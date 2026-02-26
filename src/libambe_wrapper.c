#include <mbelib-neo/mbelib.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static const int rW[36] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                           0, 1, 0, 1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2};
static const int rX[36] = {23, 10, 22, 9, 21, 8,  20, 7, 19, 6, 18, 5, 17, 4, 16, 3, 15, 2,
                           14, 1,  13, 0, 12, 10, 11, 9, 10, 8, 9,  7, 8,  6, 7,  5, 6,  4};
static const int rY[36] = {0, 2, 0, 2, 0, 2, 0, 2, 0, 3, 0, 3, 1, 3, 1, 3, 1, 3,
                           1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3};
static const int rZ[36] = {5,  3, 4,  2, 3,  1, 2,  0, 1,  13, 0,  12, 22, 11, 21, 10, 20, 9,
                           19, 8, 18, 7, 17, 6, 16, 5, 15, 4,  14, 3,  13, 2,  12, 1,  11, 0};

static const int thumbdv_inverse_map[49] = {
     0, 18, 36,  1, 19, 37,  2, 20, 38,
     3, 21, 39,  4, 22, 40,  5, 23, 41,
     6, 24, 42,  7, 25, 43,  8, 26, 44,
     9, 27, 45, 10, 28, 46, 11, 29, 47,
    12, 30, 48, 13, 31, 14, 32, 15, 33,
    16, 34, 17, 35
};

static const int thumbdv_map[49] = {
     0, 18, 36,  1, 19, 37,  2, 20, 38,
     3, 21, 39,  4, 22, 40,  5, 23, 41,
     6, 24, 42,  7, 25, 43,  8, 26, 44,
     9, 27, 45, 10, 28, 46, 11, 29, 47,
    12, 30, 48, 13, 31, 14, 32, 15, 33,
    16, 34, 17, 35
};

typedef struct {
    mbe_parms cur_mp;
    mbe_parms prev_mp;
    mbe_parms prev_mp_enhanced;
} DecoderContext;

DecoderContext* create_context() {
    DecoderContext* ctx = (DecoderContext*)malloc(sizeof(DecoderContext));
    if (ctx) {
        mbe_initMbeParms(&ctx->cur_mp, &ctx->prev_mp, &ctx->prev_mp_enhanced);
    }
    return ctx;
}

void free_context(DecoderContext* ctx) {
    if (ctx) free(ctx);
}

static void unpack_bits(const uint8_t* input_bytes, int num_bytes, char* output_bits, int num_bits) {
    int bit_count = 0;
    for (int i = 0; i < num_bytes; i++) {
        for (int j = 7; j >= 0; j--) {
            if (bit_count < num_bits) {
                output_bits[bit_count++] = (input_bytes[i] >> j) & 1;
            }
        }
    }
}

void fec_demod_wrapper(const uint8_t* in_buf_3600, char* out_ambe_d) {
    char ambe_bits[72];
    char ambe_fr[4][24];
    char temp[49];

    unpack_bits(in_buf_3600, 9, ambe_bits, 72);
    memset(ambe_fr, 0, sizeof(ambe_fr));
    memset(temp, 0, sizeof(temp));
    
    for (int sym_idx = 0; sym_idx < 36; sym_idx++) {
        char bit1 = ambe_bits[sym_idx * 2];
        char bit0 = ambe_bits[sym_idx * 2 + 1];
        ambe_fr[rW[sym_idx]][rX[sym_idx]] = bit1;
        ambe_fr[rY[sym_idx]][rZ[sym_idx]] = bit0;
    }

    mbe_eccAmbe3600x2450C0(ambe_fr);
    mbe_demodulateAmbe3600x2450Data(ambe_fr);
    mbe_eccAmbe3600x2450Data(ambe_fr, temp);

    for (int i = 0; i < 49; i++) {
        out_ambe_d[i] = temp[thumbdv_inverse_map[i]];
    }
}

int process_ambe2450(DecoderContext* ctx, const uint8_t* in_payload_7bytes, short* out_pcm) {
    if (!ctx) return 0;
    char raw_bits[49];
    char ambe_d[49];
    float float_buf[160];
    char err_str[128] = {0};
    int errs = 0, errs2 = 0;

    unpack_bits(in_payload_7bytes, 7, raw_bits, 49);
    memset(ambe_d, 0, sizeof(ambe_d));

    for (int src_idx = 0; src_idx < 49; src_idx++) {
        ambe_d[thumbdv_map[src_idx]] = raw_bits[src_idx];
    }

    mbe_processAmbe2450Dataf(float_buf, &errs, &errs2, err_str, ambe_d, 
                             &ctx->cur_mp, &ctx->prev_mp, &ctx->prev_mp_enhanced, 3);

    for (int i = 0; i < 160; i++) {
        float val = float_buf[i];
        if (val > 32760.0f) val = 32760.0f;
        if (val < -32760.0f) val = -32760.0f;
        out_pcm[i] = (short)val;
    }
    return 160;
}

int process_ambe3600(DecoderContext* ctx, const uint8_t* in_payload_9bytes, short* out_pcm) {
    if (!ctx) return 0;
    char ambe_bits[72];
    char ambe_d[49] = {0};
    char ambe_fr[4][24];
    float float_buf[160];
    char err_str[128] = {0};
    int errs = 0, errs2 = 0;

    unpack_bits(in_payload_9bytes, 9, ambe_bits, 72);
    memset(ambe_fr, 0, sizeof(ambe_fr));

    for (int sym_idx = 0; sym_idx < 36; sym_idx++) {
        char bit1 = ambe_bits[sym_idx * 2];
        char bit0 = ambe_bits[sym_idx * 2 + 1];
        ambe_fr[rW[sym_idx]][rX[sym_idx]] = bit1;
        ambe_fr[rY[sym_idx]][rZ[sym_idx]] = bit0;
    }

    mbe_processAmbe3600x2450Framef(float_buf, &errs, &errs2, err_str, ambe_fr, ambe_d,
                                   &ctx->cur_mp, &ctx->prev_mp, &ctx->prev_mp_enhanced, 3);

    for (int i = 0; i < 160; i++) {
        float val = float_buf[i];
        if (val > 32760.0f) val = 32760.0f;
        if (val < -32760.0f) val = -32760.0f;
        out_pcm[i] = (short)val;
    }
    return 160;
}