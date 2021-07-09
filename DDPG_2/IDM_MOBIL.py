class idm_mobil():
    def __init__(self, t, d_0, T, v5, b, a3, a4, a5, x4, x5, an_af, an_be, p, ao_af, ao_be, a_th, a_max, v4, v_ex, theta, delta_d, a, d, steer, clock, t0):
        self.t = t
        self.d_0 = d_0
        self.T = T
        self.v5 = v5
        self.b = b
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5
        self.x4 = x4
        self.x5 = x5
        self.an_af = an_af
        self.an_be = an_be
        self.p = p
        self.ao_af = ao_af
        self.ao_be = ao_be
        self.a_th = a_th
        self.a_max = a_max
        self.v4 = v4
        self.v_ex = v_ex
        self.theta = theta
        self.delta_d = delta_d
        self.a = a
        self.d = d
        self.steer = steer
        self.clock = clock
        self.t0 = t0
    def panduan(self):
        delta_t = 3.5
        ac_be = self.a4
        vc_af = self.v4 + self.a4*self.t
        dleta_v = self.v5-self.v4
        dleta_cv_af = self.v5+self.a5*self.t-vc_af
        an_af = self.an_af+self.an_be+self.a3*self.t
        dleta_cd = self.x5+self.v5*self.t+0.5*self.a5*self.t**2-(self.x4+self.v4*self.t+0.5*self.a4*self.t**2)
        dc_ex = self.d_0+self.T*vc_af+(vc_af*dleta_cv_af/(2*(self.a_max*self.b)**0.5))
        ac_af = self.a_max*(1-(vc_af/self.v_ex)**self.theta-(dc_ex/dleta_cd)**2)
        d_ex = self.d_0+self.T*self.v4+(self.v4*dleta_v/(2*(self.a_max*self.b)**0.5))
        if self.clock-self.t0 > delta_t:
            if ac_af-ac_be+self.p*(an_af-self.an_be+self.ao_af-self.ao_be) >= self.a_th:
                if self.steer==self.a:
                    self.steer=self.d
                    acce = 0
                    self.t0=self.clock
                else:
                    self.steer=self.a
                    acce = 0
                    self.t0=self.clock
            else:
                acce = self.a_max*(1-(self.v4/self.v_ex)**self.theta-(d_ex/self.delta_d)**2)
        else:
            acce = 0
            # acce = a_max*(1-(v4/v_ex)^theta-(d_ex/delta_d)**2)
        return self.steer,acce,self.t0