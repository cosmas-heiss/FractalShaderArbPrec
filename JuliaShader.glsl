#version 430
layout(local_size_x = 32, local_size_y = 32) in;


// REPLACE THIS LINE BY CONST N_INTS DEF //


layout(std430, binding = 0) coherent buffer iter_buffer {float table[];} iter_buf;
layout(std430, binding = 1) coherent buffer z_buffer {uint table[];} z_buf;
layout(std430, binding = 2) coherent buffer c_buffer {uint table[2 * n_ints + 2];} c_buf;


uniform ivec2 image_size;
uniform int num_iter;


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


void mult_mantisse(out uint[n_ints] out_num, in uint[n_ints] a_num, in uint[n_ints] b_num) {
    for (int i=0; i<n_ints; i++) {
        out_num[i] = 0;
    }

    uint extra_num, extra_num1, extra_num2;
    for (int i=0; i<n_ints; i++) {
        for (int j=0; j<n_ints - i; j++) {
            extra_num = a_num[i] * b_num[j];

            extra_num1 = extra_num >> 16;
            extra_num2 = extra_num & uint(65535);

            
            if (i + j >= 1) {
                out_num[i + j - 1] += extra_num1;
            }
            out_num[i + j] += extra_num2;
        }
    }

    uint carry = 0;
    for (int i=n_ints-1; i>0; i--) {
        carry = out_num[i] >> 16;
        out_num[i] &= uint(65535);
        out_num[i - 1] += carry;
    }
}


void mult(out bool out_sign, out uint[n_ints] out_num, in bool a_sign, in uint[n_ints] a_num, in bool b_sign, in uint[n_ints] b_num) {
    mult_mantisse(out_num, a_num, b_num);
    out_sign = a_sign ^^ b_sign;
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


void multx2_bitshift(inout uint[n_ints] a_num) {
    uint carry = 0;
    for (int i=n_ints-1; i>=0; i--) {
        a_num[i] = a_num[i] << 1;
        a_num[i] += carry;
        carry = a_num[i] >> 16;
        a_num[i] &= uint(65535);
    }
}


void do_iteration(inout float iter, inout bool z_real_sign, inout uint[n_ints] z_real_num, inout bool z_imag_sign,  inout uint[n_ints] z_imag_num,
                  in bool c_real_sign, in uint[n_ints] c_real_num, in bool c_imag_sign, in uint[n_ints] c_imag_num) {

    bool z_real_sq_sign;
    bool z_imag_sq_sign;
    bool z_real_x_z_imag_sign;
    
    uint[n_ints] z_real_sq_num;
    uint[n_ints] z_imag_sq_num;
    uint[n_ints] z_real_x_z_imag_num;

    float cur_rad_sq;

    for (int i=0; i<num_iter; i++) {
        mult(z_real_sq_sign, z_real_sq_num, z_real_sign, z_real_num, z_real_sign, z_real_num);
        mult(z_imag_sq_sign, z_imag_sq_num, z_imag_sign, z_imag_num, z_imag_sign, z_imag_num);
        mult(z_real_x_z_imag_sign, z_real_x_z_imag_num, z_real_sign, z_real_num, z_imag_sign, z_imag_num);

        cur_rad_sq = float(z_real_sq_num[0]) + float(z_real_sq_num[1]) * 0.00001525878 + float(z_imag_sq_num[0]) + float(z_imag_sq_num[1]) * 0.00001525878;
        if (cur_rad_sq > 250.0) {
            if (i > 0) {
                iter += 1.0 - 1.44 * log(log(cur_rad_sq) * 0.41702461128); //0.41702461128 = 1 / log(250) // 1.44 found empirically...
            }
            break;
        }

        add(z_real_sq_sign, z_real_sq_num, true, z_imag_sq_num);
        multx2_bitshift(z_real_x_z_imag_num);

        z_real_sign = z_real_sq_sign;
        z_real_num = z_real_sq_num;

        z_imag_sign = z_real_x_z_imag_sign;
        z_imag_num = z_real_x_z_imag_num;

        add(z_real_sign, z_real_num, c_real_sign, c_real_num);
        add(z_imag_sign, z_imag_num, c_imag_sign, c_imag_num);

        iter += 1.0;
    }
}


void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);

    if (pixel_coords.x < image_size.x && pixel_coords.y < image_size.y) {
        int coord = pixel_coords.x * image_size.y + pixel_coords.y;

        uint[n_ints] c_real_num, c_imag_num, z_real_num, z_imag_num;

        bool c_real_sign = c_buf.table[0] == 1;
        bool c_imag_sign = c_buf.table[n_ints + 1] == 1;

        bool z_real_sign = z_buf.table[coord * (n_ints + 1) * 2] == 1;
        bool z_imag_sign = z_buf.table[coord * (n_ints + 1) * 2 + (n_ints + 1)] == 1;

        for (int i=0; i<n_ints; i++) {
            c_real_num[i] = c_buf.table[i + 1];
            c_imag_num[i] = c_buf.table[n_ints + i + 2];

            z_real_num[i] = z_buf.table[coord * (n_ints + 1) * 2 + i + 1];
            z_imag_num[i] = z_buf.table[coord * (n_ints + 1) * 2 + (n_ints + 1) + i + 1];
        }

        float iter = iter_buf.table[coord];

        do_iteration(iter, z_real_sign, z_real_num, z_imag_sign, z_imag_num, c_real_sign, c_real_num, c_imag_sign, c_imag_num);

        z_buf.table[coord * (n_ints + 1) * 2] = z_real_sign ? 1: 0;
        z_buf.table[coord * (n_ints + 1) * 2 + (n_ints + 1)] = z_imag_sign ? 1: 0;
        for (int i=0; i<n_ints; i++) {
            z_buf.table[coord * (n_ints + 1) * 2 + i + 1] = z_real_num[i];
            z_buf.table[coord * (n_ints + 1) * 2 + (n_ints + 1) + i + 1] = z_imag_num[i];
        }

        iter_buf.table[coord] = iter;
    }
}
