import numpy as np
import moderngl
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import pygame
from decimal import getcontext, Decimal
getcontext().prec = 256
from ColorSchemes import color_palette_blue


def decimal_to_uint_ar(decimal, n_ints):
    num_sign = 1 if (decimal < 0) else 0
    decimal = abs(decimal)

    decimal = decimal * Decimal(65536**(n_ints-1))
    b = Decimal(65536)

    num = np.zeros(n_ints + 1, dtype=np.uint32)
    num[0] = num_sign

    for i in range(n_ints):
        num[n_ints - i] = int(decimal % b)
        decimal = decimal // b

    return num


class GUI():
    def __init__(self, size_x, size_y, param_bar_height=72):
        assert size_x % 2 == 0 and param_bar_height % 2 == 0
        self.size = (size_x, size_y)
        self.fractal_size = (size_x // 2, size_y - param_bar_height)
        self.param_bar_height = param_bar_height
        self.shader_iterator = ShaderIterator(*self.fractal_size)

        self.gameDisplay = pygame.display.set_mode(self.size)
        pygame.display.set_caption('Fractal Dockin Fonkey Inc.')

        pygame.font.init()
        self.relevant_digits = int((self.shader_iterator.num_prec_ints - 1) * np.log10(2) * 16)
        self.param_bar_surface = pygame.Surface((size_x, param_bar_height))
        self.font = pygame.font.SysFont('Courier New', 14)

        self.mouse_pos = (0, 0)
        self.mouse_right_pressed = False
        self.mouse_left_pressed = False

        self.mandelbrot_center = (Decimal(0), Decimal(0))
        self.julia_center = (Decimal(0), Decimal(0))
        self.mandelbrot_x_scale = 4.0
        self.julia_x_scale = 4.0
        self.julia_c = (Decimal(0), Decimal(0.8))

        self.shader_iterator.set_julia_c(self.julia_c)
        self.shader_iterator.set_mandelbrot_params(self.mandelbrot_center, self.mandelbrot_x_scale)
        self.shader_iterator.set_julia_params(self.julia_center, self.julia_x_scale)


    def decimal_to_string(self, dec_num):
        num_str = str(dec_num)
        if '.' in num_str:
            a, b = num_str.split('.')
            b = b[:self.relevant_digits]
            num_digits = len(b)
            num_str = a + '.' + b
        else:
            num_str = num_str + '.'
            num_digits = 0
        num_str = num_str + '0' * (self.relevant_digits - num_digits)
        if num_str[0] != '-':
            num_str = ' ' + num_str
        return num_str

    def draw_info(self):
        self.param_bar_surface.fill((0, 0, 0))
        julia_x_str = self.decimal_to_string(self.shader_iterator.parameter_dict['julia_center'][0])
        self.param_bar_surface.blit(self.font.render(' x:  ' + julia_x_str, 14, (255, 255, 255)), (0, 0))
        julia_y_str = self.decimal_to_string(self.shader_iterator.parameter_dict['julia_center'][1])
        self.param_bar_surface.blit(self.font.render(' y:  ' + julia_y_str, 14, (255, 255, 255)), (0, 14))
        julia_cx_str = self.decimal_to_string(self.shader_iterator.parameter_dict['julia_c'][0])
        self.param_bar_surface.blit(self.font.render(' cx: ' + julia_cx_str, 14, (255, 255, 255)), (0, 38))
        julia_cy_str = self.decimal_to_string(self.shader_iterator.parameter_dict['julia_c'][1])
        self.param_bar_surface.blit(self.font.render(' cy: ' + julia_cy_str, 14, (255, 255, 255)), (0, 52))
        julia_scale_str = '{:g}'.format(self.shader_iterator.parameter_dict['julia_x_scale'])
        self.param_bar_surface.blit(self.font.render(' scale:  ' + julia_scale_str, 14, (255, 255, 255)), (650, 0))

        julia_max_iter_str = str(self.shader_iterator.parameter_dict['julia_cur_max_iter'])
        ju_max_it_col = (0, 255, 0) if self.shader_iterator.parameter_dict['julia_all_new'] else (255, 0, 0)
        self.param_bar_surface.blit(self.font.render(' max iter:  ' + julia_max_iter_str, 14, ju_max_it_col), (650, 14))

        mandelbrot_x_str = self.decimal_to_string(self.shader_iterator.parameter_dict['mandelbrot_center'][0])
        self.param_bar_surface.blit(self.font.render(' x:  ' + mandelbrot_x_str, 14, (255, 255, 255)), (self.fractal_size[0], 0))
        mandelbrot_y_str = self.decimal_to_string(self.shader_iterator.parameter_dict['mandelbrot_center'][1])
        self.param_bar_surface.blit(self.font.render(' y:  ' + mandelbrot_y_str, 14, (255, 255, 255)), (self.fractal_size[0], 14))
        mandelbrot_scale_str = '{:g}'.format(self.shader_iterator.parameter_dict['mandelbrot_x_scale'])
        self.param_bar_surface.blit(self.font.render(' scale:  ' + mandelbrot_scale_str, 14, (255, 255, 255)), (self.fractal_size[0] + 650, 0))

        mandelbrot_max_iter_str = str(self.shader_iterator.parameter_dict['mandelbrot_cur_max_iter'])
        mb_max_it_col = (0, 255, 0) if self.shader_iterator.parameter_dict['mandelbrot_all_new'] else (255, 0, 0)
        self.param_bar_surface.blit(self.font.render(' max iter:  ' + mandelbrot_max_iter_str, 14, mb_max_it_col), (self.fractal_size[0] + 650, 14))

        self.gameDisplay.blit(self.param_bar_surface, (0, 0))


    def mouse_pos_in_julia_fractal(self):
        if self.mouse_pos[0] <= self.size[0] / 2 and self.mouse_pos[1] > self.param_bar_height:
            return True
        else:
            return False

    def mouse_pos_in_mandelbrot_fractal(self):
        if self.mouse_pos[0] > self.size[0] / 2 and self.mouse_pos[1] > self.param_bar_height:
            return True
        else:
            return False

    def pixel_to_tex_coords(self, x, y):
        if self.mouse_pos_in_julia_fractal():
            x, y = x - self.fractal_size[0] / 2, y - self.param_bar_height - self.fractal_size[1] / 2
            return (x / self.fractal_size[0], -y / self.fractal_size[0])
        elif self.mouse_pos_in_mandelbrot_fractal():
            x, y = x - self.size[0] // 2 - self.fractal_size[0] / 2, y - self.param_bar_height - self.fractal_size[1] / 2
            return (x / self.fractal_size[0], -y / self.fractal_size[0])
        else:
            return (None, None)

    def get_fractal_coords(self, x, y):
        a, b = self.pixel_to_tex_coords(x, y)
        if self.mouse_pos_in_julia_fractal():
            return (Decimal(a * self.julia_x_scale) + self.julia_center[0],
                    Decimal(b * self.julia_x_scale) + self.julia_center[1])
        elif self.mouse_pos_in_mandelbrot_fractal():
            return (Decimal(a * self.mandelbrot_x_scale) + self.mandelbrot_center[0],
                    Decimal(b * self.mandelbrot_x_scale) + self.mandelbrot_center[1])
        else:
            return (None, None)

    def scroll(self, event):
        alpha = 0.9**event.y

        if self.mouse_pos_in_julia_fractal():
            self.draw_mandelbrot = False
            a, b = self.pixel_to_tex_coords(*self.mouse_pos)
            self.julia_center = (self.julia_center[0] + Decimal(a * (1 - alpha) * self.julia_x_scale),
                                 self.julia_center[1] + Decimal(b * (1 - alpha) * self.julia_x_scale))
            self.julia_x_scale *= alpha
            self.shader_iterator.set_julia_params(self.julia_center, self.julia_x_scale)
            
        elif self.mouse_pos_in_mandelbrot_fractal():
            self.draw_julia = False
            a, b = self.pixel_to_tex_coords(*self.mouse_pos)
            self.mandelbrot_center = (self.mandelbrot_center[0] + Decimal(a * (1 - alpha) * self.mandelbrot_x_scale),
                                      self.mandelbrot_center[1] + Decimal(b * (1 - alpha) * self.mandelbrot_x_scale))
            self.mandelbrot_x_scale *= alpha
            self.shader_iterator.set_mandelbrot_params(self.mandelbrot_center, self.mandelbrot_x_scale)


    def move_centers_along_pixels(self, shift):
        if self.mouse_pos_in_julia_fractal():
            self.draw_mandelbrot = False
            self.julia_center = (self.julia_center[0] - Decimal(shift[0] * self.julia_x_scale / self.fractal_size[0]),
                                 self.julia_center[1] + Decimal(shift[1] * self.julia_x_scale / self.fractal_size[0]))
            self.shader_iterator.set_julia_params(self.julia_center, self.julia_x_scale)

        elif self.mouse_pos_in_mandelbrot_fractal():
            self.draw_julia = False
            self.mandelbrot_center = (self.mandelbrot_center[0] - Decimal(shift[0] * self.mandelbrot_x_scale / self.fractal_size[0]),
                                      self.mandelbrot_center[1] + Decimal(shift[1] * self.mandelbrot_x_scale / self.fractal_size[0]))
            self.shader_iterator.set_mandelbrot_params(self.mandelbrot_center, self.mandelbrot_x_scale)


    def mouse_down_event(self, event):
        if event.button == 3:
            self.mouse_right_pressed = True
        elif event.button == 1:
            self.mouse_left_pressed = True
            if self.mouse_pos_in_mandelbrot_fractal():
                self.julia_c = self.get_fractal_coords(*self.mouse_pos)
                self.shader_iterator.set_julia_c(self.julia_c)
            

    def mouse_up_event(self, event):
        if event.button == 3:
            self.mouse_right_pressed = False
        elif event.button == 1:
            self.mouse_left_pressed = False


    def mouse_motion_event(self, event):
        self.mouse_pos = event.pos
        if self.mouse_right_pressed:
            self.move_centers_along_pixels(event.rel)
        if self.mouse_left_pressed:
            if self.mouse_pos_in_mandelbrot_fractal():
                self.draw_mandelbrot = False
                self.julia_c = self.get_fractal_coords(*self.mouse_pos)
                self.shader_iterator.set_julia_c(self.julia_c)


    def window_leave(self, event):
        self.mouse_right_pressed = False
        self.mouse_left_pressed = False

    def mainloop(self):
        crashed = False

        while not crashed:
            self.draw_mandelbrot, self.draw_julia = True, True

            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.mouse_down_event(event)
                if event.type == pygame.MOUSEBUTTONUP:
                    self.mouse_up_event(event)
                if event.type == pygame.MOUSEMOTION:
                    self.mouse_motion_event(event)
                if event.type == pygame.WINDOWLEAVE:
                    self.window_leave(event)
                if event.type == pygame.MOUSEWHEEL:
                    self.scroll(event)
                if event.type == pygame.QUIT:
                    crashed = True
                    break

            julia_fractal, mandelbrot_fractal = self.shader_iterator.draw(draw_mandelbrot=self.draw_mandelbrot, draw_julia=self.draw_julia)

            julia_fractal_surf = pygame.pixelcopy.make_surface(julia_fractal)
            self.gameDisplay.blit(julia_fractal_surf, (0, self.param_bar_height))

            mandelbrot_fractal_surf = pygame.pixelcopy.make_surface(mandelbrot_fractal)
            self.gameDisplay.blit(mandelbrot_fractal_surf, (self.size[0] // 2, self.param_bar_height))

            self.draw_info()

            pygame.display.update()

        pygame.quit()




class ShaderIterator():
    def __init__(self, width, height, group_size=32, num_prec_ints=12, iter_steps=1000):
        self.num_prec_ints = num_prec_ints
        self.context = moderngl.create_context(standalone=True, require=430)

        self.size = (width, height)
        self.num_groups = (int(np.ceil(width / group_size)), int(np.ceil(height / group_size)))
        self.iter_steps = iter_steps

        self.program_cinit = self.context.compute_shader(self.load_file('CInitShader.glsl'))
        self.program_mandelbrot = self.context.compute_shader(self.load_file('MandelbrotShader.glsl'))
        self.program_julia = self.context.compute_shader(self.load_file('JuliaShader.glsl'))
        self.program_coloring = self.context.compute_shader(self.load_file('ColoringShader.glsl'))

        self.parameter_dict = {'mandelbrot_center': (Decimal(0), Decimal(0)),
                               'mandelbrot_x_scale': 4,
                               'mandelbrot_rel_old_shift': (0.0, 0.0),
                               'mandelbrot_rel_old_scale': 1.0,
                               'julia_center': (Decimal(0), Decimal(0)),
                               'julia_rel_old_shift': (0.0, 0.0),
                               'julia_rel_old_scale': 1.0,
                               'julia_x_scale': 4,
                               'julia_c' : (Decimal(0), Decimal(0.8)),
                               'mandelbrot_cur_max_iter': 0,
                               'julia_cur_max_iter': 0,
                               'julia_last_max_iter': 0,
                               'mandelbrot_last_max_iter': 0,
                               'mandelbrot_all_new': True,
                               'julia_all_new': True}

        self.set_up_buffers()
        self.set_up_uniforms()


    def load_file(self, path):
        with open(path, 'r') as f:
            out = f.read()
        out = out.replace('// REPLACE THIS LINE BY CONST N_INTS DEF //', 'const int n_ints = {};'.format(self.num_prec_ints))
        return out


    def set_julia_c(self, c):
        self.parameter_dict['julia_c'] = c
        self.parameter_dict['julia_rel_old_shift'] = (10000.0, 10000.0)
        self.reinit_julia_buffers()


    def set_mandelbrot_params(self, center, scale):
        self.parameter_dict['mandelbrot_rel_old_scale'] *= scale / self.parameter_dict['mandelbrot_x_scale']
        old_shift_x = float((center[0] - self.parameter_dict['mandelbrot_center'][0]) / Decimal(scale))
        old_shift_y = float((center[1] - self.parameter_dict['mandelbrot_center'][1]) / Decimal(self.size[1] * scale / self.size[0]))
        self.parameter_dict['mandelbrot_rel_old_shift'] = (self.parameter_dict['mandelbrot_rel_old_shift'][0] + old_shift_x,
                                                           self.parameter_dict['mandelbrot_rel_old_shift'][1] + old_shift_y)
        self.parameter_dict['mandelbrot_center'] = center
        self.parameter_dict['mandelbrot_x_scale'] = scale
        self.reinit_mandelbrot_buffers()


    def set_julia_params(self, center, scale):
        self.parameter_dict['julia_rel_old_scale'] *= scale / self.parameter_dict['julia_x_scale']
        old_shift_x = float((center[0] - self.parameter_dict['julia_center'][0]) / Decimal(scale))
        old_shift_y = float((center[1] - self.parameter_dict['julia_center'][1]) / Decimal(self.size[1] * scale / self.size[0]))
        self.parameter_dict['julia_rel_old_shift'] = (self.parameter_dict['julia_rel_old_shift'][0] + old_shift_x,
                                                      self.parameter_dict['julia_rel_old_shift'][1] + old_shift_y)
        self.parameter_dict['julia_center'] = center
        self.parameter_dict['julia_x_scale'] = scale
        self.reinit_julia_buffers()


    def set_up_uniforms(self):
        self.program_mandelbrot['image_size'] = self.size

        self.program_julia['image_size'] = self.size

        self.program_coloring['image_size'] = self.size
        self.program_coloring['max_iter'] = 0

        self.program_cinit['image_size'] = self.size


    def set_up_buffers(self):
        self.mandelbrot_center_buffer = self.context.buffer(reserve=4 * (self.num_prec_ints + 1) * 2)
        self.julia_center_buffer = self.context.buffer(reserve=4 * (self.num_prec_ints + 1) * 2)

        self.mandelbrot_c_buffer = self.context.buffer(reserve=4 * self.size[0] * self.size[1] * (self.num_prec_ints + 1) * 2)
        self.mandelbrot_z_buffer = self.context.buffer(reserve=4 * self.size[0] * self.size[1] * (self.num_prec_ints + 1) * 2)
        self.mandelbrot_iter_buffer = self.context.buffer(reserve=4 * self.size[0] * self.size[1])
        self.mandelbrot_old_iter_tex = self.context.texture(self.size, 4, dtype='f4', data=-np.ones(self.size[0] * self.size[1] * 4, dtype=np.float32))
        self.mandelbrot_old_iter_tex2 = self.context.texture(self.size, 4, dtype='f4', data=-np.ones(self.size[0] * self.size[1] * 4, dtype=np.float32))

        self.julia_c_buffer = self.context.buffer(reserve=4 * (self.num_prec_ints + 1) * 2)
        self.julia_z_buffer = self.context.buffer(reserve=4 * self.size[0] * self.size[1] * (self.num_prec_ints + 1) * 2)
        self.julia_iter_buffer = self.context.buffer(reserve=4 * self.size[0] * self.size[1])
        self.julia_old_iter_tex = self.context.texture(self.size, 4, dtype='f4', data=-np.ones(self.size[0] * self.size[1] * 4, dtype=np.float32))
        self.julia_old_iter_tex2 = self.context.texture(self.size, 4, dtype='f4', data=-np.ones(self.size[0] * self.size[1] * 4, dtype=np.float32))
        self.reinit_mandelbrot_buffers()
        self.reinit_julia_buffers()

        self.mandelbrot_output_tex = self.context.texture(self.size, 4, dtype='f4')
        self.julia_output_tex = self.context.texture(self.size, 4, dtype='f4')

        blue_color_with_alpha = np.concatenate((color_palette_blue, np.ones((len(color_palette_blue), 1), dtype=color_palette_blue.dtype)), axis=1)
        self.blue_color_scheme = self.context.texture((len(color_palette_blue), 1), 4, dtype='f4', data=blue_color_with_alpha)
        self.blue_color_scheme.repeat_x = True


    def write_mandelbrot_center_buffer(self):
        center_nums_x = decimal_to_uint_ar(self.parameter_dict['mandelbrot_center'][0], self.num_prec_ints)
        center_nums_y = decimal_to_uint_ar(self.parameter_dict['mandelbrot_center'][1], self.num_prec_ints)
        self.mandelbrot_center_buffer.write(np.concatenate((center_nums_x, center_nums_y), axis=0))


    def write_julia_center_and_c_buffer(self):
        center_nums_x = decimal_to_uint_ar(self.parameter_dict['julia_center'][0], self.num_prec_ints)
        center_nums_y = decimal_to_uint_ar(self.parameter_dict['julia_center'][1], self.num_prec_ints)
        self.julia_center_buffer.write(np.concatenate((center_nums_x, center_nums_y), axis=0))

        c_nums_x = decimal_to_uint_ar(self.parameter_dict['julia_c'][0], self.num_prec_ints)
        c_nums_y = decimal_to_uint_ar(self.parameter_dict['julia_c'][1], self.num_prec_ints)
        self.julia_c_buffer.write(np.concatenate((c_nums_x, c_nums_y), axis=0))


    def reinit_mandelbrot_buffers(self):
        self.write_mandelbrot_center_buffer()

        self.mandelbrot_iter_buffer.bind_to_storage_buffer(0)
        self.mandelbrot_z_buffer.bind_to_storage_buffer(1)
        self.mandelbrot_c_buffer.bind_to_storage_buffer(2)
        self.mandelbrot_center_buffer.bind_to_storage_buffer(3)
        
        self.program_cinit['mode'] = 0
        self.program_cinit['scale'] = self.parameter_dict['mandelbrot_x_scale']
        self.program_cinit.run(group_x=self.num_groups[0], group_y=self.num_groups[1])
        self.context.finish()

        self.parameter_dict['mandelbrot_last_max_iter'] = max(self.parameter_dict['mandelbrot_cur_max_iter'], self.parameter_dict['mandelbrot_last_max_iter'])
        self.parameter_dict['mandelbrot_cur_max_iter'] = 0
        

    def reinit_julia_buffers(self):
        self.write_julia_center_and_c_buffer()

        self.julia_iter_buffer.bind_to_storage_buffer(0)
        self.julia_z_buffer.bind_to_storage_buffer(1)
        self.julia_center_buffer.bind_to_storage_buffer(3)

        self.program_cinit['mode'] = 1
        self.program_cinit['scale'] = self.parameter_dict['julia_x_scale']
        self.program_cinit.run(group_x=self.num_groups[0], group_y=self.num_groups[1])
        self.context.finish()

        self.parameter_dict['julia_last_max_iter'] = max(self.parameter_dict['julia_cur_max_iter'], self.parameter_dict['julia_last_max_iter'])
        self.parameter_dict['julia_cur_max_iter'] = 0


    def get_iter_steps(self, cur_max_iter):
        if cur_max_iter < 200:
            return min(100, self.iter_steps)
        elif cur_max_iter < 1000:
            return min(400, self.iter_steps)
        else:
            return self.iter_steps


    def iterate_mandelbrot(self):
        self.mandelbrot_iter_buffer.bind_to_storage_buffer(0)
        self.mandelbrot_z_buffer.bind_to_storage_buffer(1)
        self.mandelbrot_c_buffer.bind_to_storage_buffer(2)

        iter_step = self.get_iter_steps(self.parameter_dict['mandelbrot_cur_max_iter'])
        self.program_mandelbrot['num_iter'] = iter_step
        self.program_mandelbrot.run(group_x=self.num_groups[0], group_y=self.num_groups[1])
        self.context.finish()
        
        self.parameter_dict['mandelbrot_cur_max_iter'] += iter_step


    def color_mandelbrot(self):
        self.mandelbrot_iter_buffer.bind_to_storage_buffer(0)
        self.blue_color_scheme.use(1)
        self.mandelbrot_output_tex.bind_to_image(2)
        self.mandelbrot_old_iter_tex.use(3)
        self.mandelbrot_old_iter_tex2.bind_to_image(4)

        self.program_coloring['old_iter_shift'] = self.parameter_dict['mandelbrot_rel_old_shift']
        self.program_coloring['old_iter_scale'] = self.parameter_dict['mandelbrot_rel_old_scale']
        self.parameter_dict['mandelbrot_rel_old_shift'] = (0.0, 0.0)
        self.parameter_dict['mandelbrot_rel_old_scale'] = 1.0

        self.program_coloring['max_iter'] = self.parameter_dict['mandelbrot_cur_max_iter']
        self.program_coloring['last_max_iter'] = self.parameter_dict['mandelbrot_last_max_iter']
        self.parameter_dict['mandelbrot_last_max_iter'] *= 0.95
        self.program_coloring.run(group_x=self.num_groups[0], group_y=self.num_groups[1])
        self.context.finish()

        old_iter_data = np.frombuffer(self.mandelbrot_old_iter_tex.read(), np.float32).reshape(-1, 4)
        self.parameter_dict['mandelbrot_all_new'] = np.all(old_iter_data[:, 2] > 0.5)

        self.mandelbrot_old_iter_tex, self.mandelbrot_old_iter_tex2 = self.mandelbrot_old_iter_tex2, self.mandelbrot_old_iter_tex


    def iterate_julia(self):
        self.julia_iter_buffer.bind_to_storage_buffer(0)
        self.julia_z_buffer.bind_to_storage_buffer(1)
        self.julia_c_buffer.bind_to_storage_buffer(2)

        iter_step = self.get_iter_steps(self.parameter_dict['julia_cur_max_iter'])
        self.program_julia['num_iter'] = iter_step
        self.program_julia.run(group_x=self.num_groups[0], group_y=self.num_groups[1])
        self.context.finish()
        
        self.parameter_dict['julia_cur_max_iter'] += iter_step


    def color_julia(self):
        self.julia_iter_buffer.bind_to_storage_buffer(0)
        self.blue_color_scheme.use(1)
        self.julia_output_tex.bind_to_image(2)
        self.julia_old_iter_tex.use(3)
        self.julia_old_iter_tex2.bind_to_image(4)

        self.program_coloring['old_iter_shift'] = self.parameter_dict['julia_rel_old_shift']
        self.program_coloring['old_iter_scale'] = self.parameter_dict['julia_rel_old_scale']
        self.parameter_dict['julia_rel_old_shift'] = (0.0, 0.0)
        self.parameter_dict['julia_rel_old_scale'] = 1.0

        self.program_coloring['max_iter'] = self.parameter_dict['julia_cur_max_iter']
        self.program_coloring['last_max_iter'] = self.parameter_dict['julia_last_max_iter']
        self.parameter_dict['julia_last_max_iter'] *= 0.95
        self.program_coloring.run(group_x=self.num_groups[0], group_y=self.num_groups[1])
        self.context.finish()

        old_iter_data = np.frombuffer(self.julia_old_iter_tex2.read(), np.float32).reshape(-1, 4)
        self.parameter_dict['julia_all_new'] = np.all(old_iter_data[:, 2] > 0.5)

        self.julia_old_iter_tex, self.julia_old_iter_tex2 = self.julia_old_iter_tex2, self.julia_old_iter_tex


    def draw_mandelbrot(self):
        self.iterate_mandelbrot()
        self.color_mandelbrot()


    def draw_julia(self):
        self.iterate_julia()
        self.color_julia()


    def draw(self, draw_mandelbrot=True, draw_julia=True):
        if draw_mandelbrot:
            self.draw_mandelbrot()
        if draw_julia:
            self.draw_julia()

        image_mandelbrot = np.frombuffer(self.mandelbrot_output_tex.read(), np.float32).reshape(self.size[1], self.size[0], 4).transpose(1, 0, 2)[:, ::-1]
        image_julia = np.frombuffer(self.julia_output_tex.read(), np.float32).reshape(self.size[1], self.size[0], 4).transpose(1, 0, 2)[:, ::-1]

        return (image_julia * 255).astype(np.uint8)[:, :, :3], (image_mandelbrot * 255).astype(np.uint8)[:, :, :3]



if __name__ == "__main__":
    iterator = GUI(1800, 900)
    iterator.mainloop()
    
    