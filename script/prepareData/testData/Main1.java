@Override
@Deprecated
int finalise(int hash, int unprocessedLength, byte[] unprocessed, int totalLen) {
    switch(unprocessedLength) {
        case 3:
            k1 ^= unprocessed[2] << 16;
        case 2:
            k1 ^= unprocessed[1] << 8;
    }
}
