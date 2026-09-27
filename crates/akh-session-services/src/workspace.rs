//! Descriptor-relative workspace IO. Symlinks are deliberately rejected, including intermediate ones.
use crate::types::{Fault, Result};
use sha2::{Digest, Sha256};
use std::{
    ffi::CString,
    fs::File,
    io::{Read, Write},
    os::fd::{AsRawFd, FromRawFd},
    path::{Component, Path, PathBuf},
};

pub const MAX_FILE_BYTES: usize = 8 * 1024 * 1024;

fn io_error(error: std::io::Error) -> Fault {
    Fault::new(
        if error.kind() == std::io::ErrorKind::NotFound {
            "not_found"
        } else {
            "workspace_io"
        },
        error.to_string(),
    )
}
pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub struct Workspace {
    root: File,
    pub path: PathBuf,
}

fn parts(path: &str) -> Result<Vec<CString>> {
    let mut parts = Vec::new();
    for component in Path::new(path).components() {
        match component {
            Component::Normal(s) => parts.push(
                CString::new(s.as_encoded_bytes())
                    .map_err(|_| Fault::invalid("NUL in workspace path"))?,
            ),
            Component::CurDir => {}
            _ => {
                return Err(Fault::invalid(
                    "workspace paths must be relative without parent traversal",
                ));
            }
        }
    }
    Ok(parts)
}

fn open_at(parent: &File, name: &CString, flags: i32) -> Result<File> {
    let fd = unsafe {
        libc::openat(
            parent.as_raw_fd(),
            name.as_ptr(),
            flags | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            0o600,
        )
    };
    if fd < 0 {
        return Err(io_error(std::io::Error::last_os_error()));
    }
    Ok(unsafe { File::from_raw_fd(fd) })
}

impl Workspace {
    pub(crate) fn from_directory(root: File, path: PathBuf) -> Self {
        Self { root, path }
    }
    pub fn open(path: &Path) -> Result<Self> {
        let path = path.canonicalize().map_err(io_error)?;
        let root = File::open(&path).map_err(io_error)?;
        if !root.metadata().map_err(io_error)?.is_dir() {
            return Err(Fault::invalid("workspace root is not a directory"));
        }
        Ok(Self { root, path })
    }
    pub fn directory(&self, path: &str) -> Result<File> {
        let mut dir = self.root.try_clone().map_err(io_error)?;
        for part in parts(path)? {
            dir = open_at(&dir, &part, libc::O_RDONLY | libc::O_DIRECTORY)?;
        }
        Ok(dir)
    }
    fn parent(&self, path: &str) -> Result<(File, CString)> {
        let mut parts = parts(path)?;
        let name = parts
            .pop()
            .ok_or_else(|| Fault::invalid("a file path is required"))?;
        let mut dir = self.root.try_clone().map_err(io_error)?;
        for part in parts {
            dir = open_at(&dir, &part, libc::O_RDONLY | libc::O_DIRECTORY)?;
        }
        Ok((dir, name))
    }
    fn read_at(dir: &File, name: &CString) -> Result<Vec<u8>> {
        let file = open_at(dir, name, libc::O_RDONLY | libc::O_NONBLOCK)?;
        let metadata = file.metadata().map_err(io_error)?;
        if !metadata.is_file() {
            return Err(Fault::invalid("workspace read requires a regular file"));
        }
        if metadata.len() > MAX_FILE_BYTES as u64 {
            return Err(Fault::invalid("file exceeds 8 MiB transfer limit"));
        }
        let mut bytes = Vec::new();
        file.take(MAX_FILE_BYTES as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(io_error)?;
        if bytes.len() > MAX_FILE_BYTES {
            return Err(Fault::invalid("file exceeds 8 MiB transfer limit"));
        }
        Ok(bytes)
    }
    pub fn read(&self, path: &str) -> Result<Vec<u8>> {
        let (dir, name) = self.parent(path)?;
        Self::read_at(&dir, &name)
    }

    /// Compare against the observed revision, then atomically replace the file in its parent directory.
    /// Other editor processes are separate writers; callers must handle revision conflicts.
    pub fn write(
        &self,
        path: &str,
        bytes: &[u8],
        expected: Option<&str>,
        create: bool,
    ) -> Result<String> {
        if bytes.len() > MAX_FILE_BYTES {
            return Err(Fault::invalid("file exceeds 8 MiB transfer limit"));
        }
        let (dir, name) = self.parent(path)?;
        let check = || -> Result<()> {
            if create {
                let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
                let rc = unsafe {
                    libc::fstatat(
                        dir.as_raw_fd(),
                        name.as_ptr(),
                        stat.as_mut_ptr(),
                        libc::AT_SYMLINK_NOFOLLOW,
                    )
                };
                if rc == 0 {
                    return Err(Fault::conflict("create would overwrite an existing entry"));
                }
                let error = std::io::Error::last_os_error();
                if error.kind() != std::io::ErrorKind::NotFound {
                    return Err(io_error(error));
                }
            } else {
                let expected = expected.ok_or_else(|| {
                    Fault::invalid("write requires expected_revision or create=true")
                })?;
                if digest(&Self::read_at(&dir, &name)?) != expected {
                    return Err(Fault::conflict("file changed since it was read"));
                }
            }
            Ok(())
        };
        check()?;
        let temp = CString::new(format!(".akh-write-{}", crate::types::new_id())).unwrap();
        let mut file = open_at(&dir, &temp, libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL)?;
        let result = (|| {
            // Preserve executable bits when replacing an existing regular file.
            if !create {
                let old = open_at(&dir, &name, libc::O_RDONLY | libc::O_NONBLOCK)?;
                use std::os::unix::fs::PermissionsExt;
                let mode = old.metadata().map_err(io_error)?.permissions().mode() & 0o777;
                file.set_permissions(std::fs::Permissions::from_mode(mode))
                    .map_err(io_error)?;
            }
            file.write_all(bytes).map_err(io_error)?;
            file.sync_all().map_err(io_error)?;
            check()?;
            if create {
                // linkat is no-replace atomically, unlike a check followed by rename.
                if unsafe {
                    libc::linkat(
                        dir.as_raw_fd(),
                        temp.as_ptr(),
                        dir.as_raw_fd(),
                        name.as_ptr(),
                        0,
                    )
                } < 0
                {
                    return Err(io_error(std::io::Error::last_os_error()));
                }
                unsafe {
                    libc::unlinkat(dir.as_raw_fd(), temp.as_ptr(), 0);
                }
            } else if unsafe {
                libc::renameat(
                    dir.as_raw_fd(),
                    temp.as_ptr(),
                    dir.as_raw_fd(),
                    name.as_ptr(),
                )
            } < 0
            {
                return Err(io_error(std::io::Error::last_os_error()));
            }
            dir.sync_all().map_err(io_error)?;
            Ok(digest(bytes))
        })();
        if result.is_err() {
            unsafe {
                libc::unlinkat(dir.as_raw_fd(), temp.as_ptr(), 0);
            }
        }
        result
    }

    pub fn mkdir(&self, path: &str) -> Result<()> {
        let mut dir = self.root.try_clone().map_err(io_error)?;
        for part in parts(path)? {
            if unsafe { libc::mkdirat(dir.as_raw_fd(), part.as_ptr(), 0o700) } < 0 {
                let error = std::io::Error::last_os_error();
                if error.kind() != std::io::ErrorKind::AlreadyExists {
                    return Err(io_error(error));
                }
            }
            dir = open_at(&dir, &part, libc::O_RDONLY | libc::O_DIRECTORY)?;
        }
        Ok(())
    }

    /// Describe one entry without following a final symlink. A missing entry,
    /// or a path through a missing directory, is `exists: false`, not an error.
    /// Intermediate symlinks are still rejected, as in every other verb.
    pub fn stat(&self, path: &str) -> Result<crate::types::WorkspaceStat> {
        use crate::types::{WorkspaceEntryKind, WorkspaceStat};
        let (dir, name) = match parts(path)?.pop() {
            None => (self.root.try_clone().map_err(io_error)?, None),
            Some(_) => match self.parent(path) {
                Ok((dir, name)) => (dir, Some(name)),
                Err(fault) if fault.code == "not_found" => return Ok(WorkspaceStat::MISSING),
                Err(fault) => return Err(fault),
            },
        };
        let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
        let rc = match &name {
            Some(name) => unsafe {
                libc::fstatat(
                    dir.as_raw_fd(),
                    name.as_ptr(),
                    stat.as_mut_ptr(),
                    libc::AT_SYMLINK_NOFOLLOW,
                )
            },
            None => unsafe { libc::fstat(dir.as_raw_fd(), stat.as_mut_ptr()) },
        };
        if rc < 0 {
            let error = std::io::Error::last_os_error();
            return match error.raw_os_error() {
                Some(libc::ENOENT) | Some(libc::ENOTDIR) => Ok(WorkspaceStat::MISSING),
                _ => Err(io_error(error)),
            };
        }
        let stat = unsafe { stat.assume_init() };
        let kind = match stat.st_mode & libc::S_IFMT {
            libc::S_IFREG => WorkspaceEntryKind::File,
            libc::S_IFDIR => WorkspaceEntryKind::Directory,
            libc::S_IFLNK => WorkspaceEntryKind::Symlink,
            _ => WorkspaceEntryKind::Other,
        };
        Ok(WorkspaceStat {
            exists: true,
            kind: Some(kind),
            size: (kind == WorkspaceEntryKind::File).then_some(stat.st_size as u64),
        })
    }

    pub fn list(&self, path: &str) -> Result<Vec<String>> {
        let dir = self.directory(path)?;
        let fd = unsafe { libc::dup(dir.as_raw_fd()) };
        if fd < 0 {
            return Err(io_error(std::io::Error::last_os_error()));
        }
        let stream = unsafe { libc::fdopendir(fd) };
        if stream.is_null() {
            unsafe { libc::close(fd) };
            return Err(io_error(std::io::Error::last_os_error()));
        }
        let mut names = Vec::new();
        loop {
            let entry = unsafe { libc::readdir(stream) };
            if entry.is_null() {
                break;
            }
            let name = unsafe { std::ffi::CStr::from_ptr((*entry).d_name.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            if name != "." && name != ".." {
                names.push(name);
            }
            if names.len() > 100_000 {
                unsafe { libc::closedir(stream) };
                return Err(Fault::invalid("directory exceeds listing limit"));
            }
        }
        unsafe { libc::closedir(stream) };
        names.sort();
        Ok(names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stat_reports_kind_and_size_without_following_symlinks() {
        use crate::types::{WorkspaceEntryKind, WorkspaceStat};
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret"), b"0123456789").unwrap();
        let ws = Workspace::open(dir.path()).unwrap();
        ws.mkdir("src").unwrap();
        ws.write("src/file", b"four", None, true).unwrap();
        std::os::unix::fs::symlink(outside.path().join("secret"), dir.path().join("link")).unwrap();
        let file = ws.stat("src/file").unwrap();
        assert_eq!(file.kind, Some(WorkspaceEntryKind::File));
        assert_eq!(file.size, Some(4));
        assert_eq!(
            ws.stat("src").unwrap().kind,
            Some(WorkspaceEntryKind::Directory)
        );
        assert_eq!(
            ws.stat(".").unwrap().kind,
            Some(WorkspaceEntryKind::Directory)
        );
        // The symlink is described, never followed to the outside file's size.
        let link = ws.stat("link").unwrap();
        assert!(link.is_symlink());
        assert_eq!(link.size, None);
        assert_eq!(ws.stat("absent").unwrap(), WorkspaceStat::MISSING);
        assert_eq!(ws.stat("absent/deeper").unwrap(), WorkspaceStat::MISSING);
        assert!(ws.stat("../escape").is_err());
    }
    #[test]
    fn revision_checks_and_atomic_create_preserve_an_editors_changes() {
        let dir = tempfile::tempdir().unwrap();
        let ws = Workspace::open(dir.path()).unwrap();
        ws.mkdir("src/nested").unwrap();
        let rev = ws.write("src/file", b"one", None, true).unwrap();
        assert!(ws.write("src/file", b"two", None, true).is_err());
        std::fs::write(dir.path().join("src/file"), b"editor").unwrap();
        assert!(ws.write("src/file", b"two", Some(&rev), false).is_err());
        assert_eq!(ws.read("src/file").unwrap(), b"editor");
        ws.write("src/file", b"two", Some(&digest(b"editor")), false)
            .unwrap();
        assert_eq!(ws.list("src").unwrap(), vec!["file", "nested"]);
    }
    #[test]
    fn traversal_and_symlink_components_never_access_outside_files() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret"), b"outside").unwrap();
        std::os::unix::fs::symlink(outside.path(), dir.path().join("link")).unwrap();
        std::os::unix::fs::symlink(outside.path().join("secret"), dir.path().join("file")).unwrap();
        let ws = Workspace::open(dir.path()).unwrap();
        for path in ["../secret", "/etc/passwd", "link/secret", "file"] {
            assert!(ws.read(path).is_err(), "{path}");
        }
        assert!(ws.write("link/secret", b"wrong", None, true).is_err());
        assert!(ws.mkdir("link/nested").is_err());
        assert_eq!(
            std::fs::read(outside.path().join("secret")).unwrap(),
            b"outside"
        );
    }
}
