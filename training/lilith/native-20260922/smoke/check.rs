fn select_le16_byte(value: u16, byte: u8) -> u8 {
    if byte == 0 { value as u8 } else { (value >> 8) as u8 }
}

fn main() {
    assert_eq!(select_le16_byte(0x1234, 0), 0x34);
    assert_eq!(select_le16_byte(0x1234, 1), 0x12);
    println!("rust-native-smoke: ok");
}
