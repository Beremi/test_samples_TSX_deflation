from tunnel_with_subdomains import tsx_setup_and_computation, prepare_coefficient_functions, load_mesh_and_domain_tags
import numpy as np


class SolverTSX():
    def __init__(self, solver_id=0, output_dir=None, solver_type='direct', rtol=None, filename=None, scale=1.0):
        self.no_parameters = 9*4
        self.no_observations = 2*18*4

        self.mesh, self.cell_tags, _ = load_mesh_and_domain_tags("tsx_ellipses_regions_coarse")
        # hardcoded number of domains, 'constant' depending on the used mesh
        self.number_of_subdomains = 9  # TODO: could be taken from the mesh
        self.alpha_values = [0.2] * self.number_of_subdomains
        self.solver_type = solver_type
        self.rtol = rtol
        self.filename = filename
        self.scale = scale

    def set_parameters(self, par):
        NS = 9
        self.permeability_values = par[:NS]
        self.storativity_values = 1/par[NS:2*NS]
        young_values = par[2*NS:3*NS] # np.array([6e10] * self.number_of_subdomains)
        poisson_values = par[3*NS:4*NS] # np.array([0.2] * self.number_of_subdomains)
        self.sigma_x = par[4*NS] # 45e6
        self.sigma_y = par[4*NS + 1] # 11e6
        self.sigma_angle = par[4*NS + 2] # 0.0
        self.mu_values = young_values / (2 * (1 + poisson_values))
        self.lmbda_values = young_values * poisson_values / ((1 + poisson_values) * (1 - 2 * poisson_values))

    def get_observations(self):
        lambda_fnc, mu_fnc, alpha_fnc, cpp_fnc, k_fnc = prepare_coefficient_functions(self.mesh, self.cell_tags,
                                                                                      self.lmbda_values, self.mu_values, self.alpha_values,
                                                                                      self.storativity_values, self.permeability_values)

        data = tsx_setup_and_computation(self.mesh,
                                         lambda_fnc, mu_fnc, alpha_fnc, cpp_fnc, k_fnc, sigma_xx=-self.sigma_x, sigma_yy=-self.sigma_y,
                                         sigma_angle=self.sigma_angle, tau_f=24*60*60/2, t_steps_num=358*2,
                                         solver_type=self.solver_type, rtol=self.rtol, filename=self.filename, scale=self.scale)
        data_fp = np.zeros((4, len(data)))
        for i, item in enumerate(data):
            data_fp[:, i] = [value[0] for value in data[i]]

        # names = ['HGT1-5', 'HGT1-4', 'HGT2-3', 'HGT2-4']
        res = []
        for i, timeline in enumerate(data_fp):
            # print(timeline)
            res.append(timeline[17*2-20::20]/9806)

        self.data = data
        res = np.array(res).reshape((-1,))
        # print("norm:", np.linalg.norm(res - observations), flush=True)
        return res

    def get_observations_for_pictures(self):
        lambda_fnc, mu_fnc, alpha_fnc, cpp_fnc, k_fnc = prepare_coefficient_functions(self.mesh, self.cell_tags,
                                                                                      self.lmbda_values, self.mu_values, self.alpha_values,
                                                                                      self.storativity_values, self.permeability_values)

        data = tsx_setup_and_computation(self.mesh,
                                         lambda_fnc, mu_fnc, alpha_fnc, cpp_fnc, k_fnc, sigma_xx=-self.sigma_x, sigma_yy=-self.sigma_y,
                                         sigma_angle=self.sigma_angle, tau_f=24*60*60/2, t_steps_num=358*2,
                                         solver_type=self.solver_type, rtol=self.rtol, filename=self.filename)
        return data


observations = [493.32752467, 754.64805252, 788.37572466, 755.01945387,
       708.27183405, 655.95978987, 614.47820502, 594.39699416,
       567.17847243, 530.43745741, 510.17716475, 515.8816575 ,
       515.30772886, 494.97253552, 477.93022495, 475.59152216,
       465.4165135 , 452.86384229, 448.7894965 , 433.26618307,
       415.84059895, 382.97842896, 365.25883128, 351.99284788,
       347.27793831, 332.17098259, 318.64508125, 309.92537921,
       306.39725322, 300.5676362 , 299.18608617, 303.92681608,
       297.3877381 , 287.90724713, 283.71612747, 291.92276435,
       381.11503398, 515.43370038, 567.87906848, 587.81872627,
       596.29976222, 600.23143393, 604.24267168, 604.48910264,
       604.59397536, 600.39683158, 597.60651299, 599.11176231,
       602.89878407, 603.23124273, 600.17454143, 599.23918313,
       598.6708189 , 599.03415368, 602.01076471, 601.54081621,
       598.76348826, 584.89274096, 576.34968972, 571.77321889,
       568.91302381, 559.71106412, 553.60537405, 551.93560232,
       551.61324337, 550.11891225, 548.41132805, 547.36679965,
       547.86213492, 546.21690982, 543.5660486 , 542.02776456,
       213.143848  , 182.31709971, 190.03880789, 199.55739055,
       206.60667528, 212.15676692, 219.18237427, 224.15477335,
       227.2746881 , 229.70971358, 234.61305484, 238.39284514,
       247.44961921, 253.75478535, 259.22186182, 262.23129018,
       266.02233652, 268.99356638, 271.63864124, 276.3197908 ,
       279.81352471, 278.0979942 , 279.34110297, 281.01668497,
       280.79661307, 280.27204155, 282.13577722, 284.22676206,
       285.1428992 , 286.21291704, 288.94563252, 290.6920895 ,
       293.00316853, 294.08053105, 293.67235285, 294.73164677,
        87.04308352,  48.08262448,  42.68103859,  42.93917678,
        45.08236787,  49.94181195,  56.0076371 ,  61.35265637,
        60.28968371,  59.6642977 ,  64.549756  ,  75.47073607,
        82.62726862,  81.53250879,  83.43589377,  90.51661055,
        91.54043429,  90.82438687,  91.61217871,  89.70334219,
        87.2880247 ,  79.27229886,  79.34111134,  79.20607144,
        82.11064079,  83.22591843,  80.31607801,  77.02257697,
        78.32361785,  78.10138551,  78.80241434,  84.91265489,
        79.66457192,  75.86837226,  77.25871781,  82.69544488]

observations = np.array(observations)
