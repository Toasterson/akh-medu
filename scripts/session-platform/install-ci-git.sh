#!/bin/sh
# Run only inside a disposable Linux CI guest. The snapshot reader requires
# --no-lazy-fetch; Ubuntu 22.04's packaged Git predates that option.
set -eu
test "$(uname -s)" = Linux
akh_git_build=$(mktemp -d)
trap 'rm -rf "$akh_git_build"' EXIT HUP INT TERM
curl --proto '=https' --tlsv1.2 -fsSL \
    https://www.kernel.org/pub/software/scm/git/git-2.55.0.tar.xz \
    -o "$akh_git_build/git.tar.xz"
printf '%s  %s\n' 457fdb04dc8728e007d4688695e6912e6f680727920f2a40bf11eacc17505357 \
    "$akh_git_build/git.tar.xz" | sha256sum -c -
tar -xJf "$akh_git_build/git.tar.xz" -C "$akh_git_build"
make -C "$akh_git_build/git-2.55.0" -j2 prefix=/usr \
    NO_GETTEXT=YesPlease NO_TCLTK=YesPlease all
sudo make -C "$akh_git_build/git-2.55.0" prefix=/usr \
    NO_GETTEXT=YesPlease NO_TCLTK=YesPlease install
/usr/bin/git --no-lazy-fetch --version
