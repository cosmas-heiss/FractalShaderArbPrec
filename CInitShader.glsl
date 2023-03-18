#version 430
#pragma optionNV unroll all
layout(local_size_x = 32, local_size_y = 32) in;


// REPLACE THIS LINE BY CONST N_INTS DEF //


layout(std430, binding = 0) coherent buffer iter_buffer {float table[];} iter_buf;
layout(std430, binding = 1) coherent buffer z_buffer {uint table[];} z_buf;
layout(std430, binding = 2) coherent buffer c_buffer {uint table[];} c_buf;
layout(std430, binding = 3) coherent buffer center_buffer {uint table[2 * n_ints + 2];} center_buf;


uniform ivec2 image_size;
uniform double scale;
uniform int mode;


void double_to_uint_ar(inout bool a_sign, inout uint[n_ints] a_num, in double b) {
    a_sign = b < 0;
    b = abs(b);
    double b2;

    uint c;
    for (int i=0; i<n_ints; i++) {
        c = uint(b);
        b -= double(c);
        b *= 65536.0;
        c &= uint(65535);

        a_num[i] = c;
    }
}


bool geq_mantisse(in uint[n_ints] a_num, in uint[n_ints] b_num) {
    uint a, b;
    for (int i=0; i<n_ints; i++) {
        a = a_num[i];
        b = b_num[i];
        if (a != b) {
            return a > b;
        }
    }
    return true;
}


void add_mantisse(inout uint[n_ints] a_num, in uint[n_ints] b_num) {
    uint carry = 0;
    for (int i=n_ints-1; i>=0; i--) {
        a_num[i] += b_num[i] + carry;
        carry = a_num[i] >> 16;
        a_num[i] &= uint(65535);
    }
}


void subtract_mantisse_big_small(inout uint[n_ints] a_num, in uint[n_ints] b_num) {
    uint carry = 0;
    uint subtractor = 0;
    for (int i=n_ints-1; i>=0; i--) {
        subtractor = b_num[i] + carry;
        if (a_num[i] >= subtractor) {
            a_num[i] -= subtractor;
            carry = 0;
        } else {
            a_num[i] += uint(65536) - subtractor;
            carry = 1;
        }
    }
}


void add(inout bool a_sign, inout uint[n_ints] a_num, in bool b_sign, in uint[n_ints] b_num) {
    if (a_sign == b_sign) {
        add_mantisse(a_num, b_num);
    } else {
        if (geq_mantisse(a_num, b_num)) {
            subtract_mantisse_big_small(a_num, b_num);
        } else {
            subtract_mantisse_big_small(b_num, a_num);
            a_num = b_num;
            a_sign = b_sign;
        }
    }
}


void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);

    if (pixel_coords.x < image_size.x && pixel_coords.y < image_size.y) {
        int coord = pixel_coords.x * image_size.y + pixel_coords.y;
        dvec2 offset = dvec2((pixel_coords.x + 0.5) / image_size.x - 0.5, (pixel_coords.y + 0.5) / image_size.x - 0.5 * image_size.y / image_size.x) * scale;

        bool c_real_sign, c_imag_sign, center_real_sign, center_imag_sign;
        uint[n_ints] c_real_num, c_imag_num, center_real_num, center_imag_num;

        double_to_uint_ar(c_real_sign, c_real_num, offset.x);
        double_to_uint_ar(c_imag_sign, c_imag_num, offset.y);

        center_real_sign = center_buf.table[0] == 1;
        center_imag_sign = center_buf.table[n_ints + 1] == 1;
        for (int i=0; i<n_ints; i++) {
            center_real_num[i] = center_buf.table[i + 1];
            center_imag_num[i] = center_buf.table[n_ints + i + 2];
        }

        
        add(c_real_sign, c_real_num, center_real_sign, center_real_num);
        add(c_imag_sign, c_imag_num, center_imag_sign, center_imag_num);
        

        z_buf.table[coord * (n_ints + 1) * 2] = c_real_sign ? 1: 0;
        z_buf.table[coord * (n_ints + 1) * 2 + (n_ints + 1)] = c_imag_sign ? 1: 0;
        for (int i=0; i<n_ints; i++) {
            z_buf.table[coord * (n_ints + 1) * 2 + i + 1] = c_real_num[i];
            z_buf.table[coord * (n_ints + 1) * 2 + (n_ints + 1) + i + 1] = c_imag_num[i];
        }

        iter_buf.table[coord] = 0.0;

        // for the mandelbrot case, c has to be initialized the same
        if (mode == 0) {
            c_buf.table[coord * (n_ints + 1) * 2] = c_real_sign ? 1: 0;
            c_buf.table[coord * (n_ints + 1) * 2 + (n_ints + 1)] = c_imag_sign ? 1: 0;
            for (int i=0; i<n_ints; i++) {
                c_buf.table[coord * (n_ints + 1) * 2 + i + 1] = c_real_num[i];
                c_buf.table[coord * (n_ints + 1) * 2 + (n_ints + 1) + i + 1] = c_imag_num[i];
            }
        }

    }
}
