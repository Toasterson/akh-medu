#include <stdint.h>
#include <stdio.h>

static uint16_t compose_le16(uint8_t low, uint8_t high) {
    return (uint16_t)low | ((uint16_t)high << 8);
}

int main(void) {
    if (compose_le16(0x34, 0x12) != 0x1234) return 1;
    puts("c-native-smoke: ok");
    return 0;
}
