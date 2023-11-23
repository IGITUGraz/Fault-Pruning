class Memristor:
    def __init__(self, w_max, noise_level=0.):
        # id_mem = 3
        self.R_min = 6843.97
        self.R_max = 14109.06

        self.R_min = (1 + noise_level) * self.R_min
        self.R_max = (1 - noise_level) * self.R_max

        self.Rc = 2. * (self.R_min * self.R_max) / (self.R_min + self.R_max)
        self.alpha = w_max / (1. / self.R_min - 1. / self.Rc)

    def get_attributes(self):
        return self.R_min, self.R_max, self.alpha, self.Rc

    def get_attributes_dict(self):
        att_dct = {'R_min': self.R_min,
                   'R_max': self.R_max,
                   'alpha': self.alpha,
                   'Rc': self.Rc
                   }
        return att_dct
