# %%
import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
import nlopt
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import pickle

# %%
mp.verbosity(1)
um_scale = 1
seed = 240  # setting the random number seed as 240 (for reproducibility)
np.random.seed(seed)  # numpy random seed setting

# Material setting and refractive index setting
def def_WS2():

    # default unit length is 1 μm
    um_scale = 1.0

    # conversion factor for eV to 1/μm [=1/hc]
    eV_um_scale = um_scale / 1.23984193

    # ------------------------------------------------------------------
    # WS2 bulk from  https://doi.org/10.1021/acs.nanolett.3c02051
    # wavelength range: 0.6 - 0.8 μm
    # Material modeling code is in GitHub
    WS2_o_eps_infty = 10.5686191135809
    WS2_o_frq1 = 1.60571176600312
    WS2_o_gamma1 = 0.0483718190665989
    WS2_o_sig1 = 0.109917872778838
    WS2_o_frq2 = 1.45174790445326
    WS2_o_gamma2 = 0.239930959161920
    WS2_o_sig2 = 0.0856209686274106
    WS2_o_frq3 = 1.96136652563532
    WS2_o_gamma3 = 1.08966976489645e-12
    WS2_o_sig3 = 3.37987155948019
    WS2_o_frq4 = 1.58619498393870
    WS2_o_gamma4 = 0.0329518082904728
    WS2_o_sig4 = 0.254224343363809
    WS2_o_frq5 =  1.66231438960124
    WS2_o_gamma5 = 0.100006254332917
    WS2_o_sig5 = 0.203902073861584
    
    WS2_e_eps_infty = 1
    WS2_e_frq1 =  3.55003330756111
    WS2_e_sig1 =  4.78000849186307
    WS2_e_gamma1 = 0

    WS2_susc = [
        mp.LorentzianSusceptibility(frequency=WS2_o_frq1, gamma=WS2_o_gamma1, sigma_diag=WS2_o_sig1*mp.Vector3(1,1,0)),
        mp.LorentzianSusceptibility(frequency=WS2_o_frq2, gamma=WS2_o_gamma2, sigma_diag=WS2_o_sig2*mp.Vector3(1,1,0)),
        mp.LorentzianSusceptibility(frequency=WS2_o_frq3, gamma=WS2_o_gamma3, sigma_diag=WS2_o_sig3*mp.Vector3(1,1,0)),
        mp.LorentzianSusceptibility(frequency=WS2_o_frq4, gamma=WS2_o_gamma4, sigma_diag=WS2_o_sig4*mp.Vector3(1,1,0)),
        mp.LorentzianSusceptibility(frequency=WS2_o_frq5, gamma=WS2_o_gamma5, sigma_diag=WS2_o_sig5*mp.Vector3(1,1,0)),
        mp.LorentzianSusceptibility(frequency=WS2_e_frq1, gamma=WS2_e_gamma1, sigma_diag=WS2_e_sig1*mp.Vector3(0,0,1)),
    ]

    WS2_material = mp.Medium(epsilon_diag=mp.Vector3(WS2_o_eps_infty,WS2_o_eps_infty, WS2_e_eps_infty), E_susceptibilities=WS2_susc)
    return WS2_material
# Ge = mp.Medium(index=4.2)
Glass = mp.Medium(index=1.5)

Air = mp.Medium(epsilon_diag=mp.Vector3(1,1,1))

WS2_material = def_WS2()

# simulation space setting
design_region_height = 0.25
design_region_width = 0.38
lpml = 0.5
Lx = design_region_width
Ly = design_region_width
Lz = 2*lpml + design_region_height + 1.0
cell_size = mp.Vector3(Lx, Ly, Lz)
resolution = 100
sim_time = 40
folder_name = "geometry_400" #location for saving the results 

# wavelength, frequency setting
wavelength = 0.680
frequency = 1/wavelength
width = 0.5

# source setting
source_pos = 0.5*Lz - lpml - 1.0/resolution
source_center = mp.Vector3(0, 0 , source_pos) # Source position
source_size = mp.Vector3(Lx , Ly, 0)

top_monitor_pos, bot_monitor_pos = +0.5*Lz-lpml-0.1, -0.5*Lz+lpml+0.1


src = mp.GaussianSource(frequency=frequency, fwidth=frequency*width, is_integrated=True)

source_L = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center, amplitude = 1),
          mp.Source(src, component=mp.Ey, size=source_size, center=source_center, amplitude = -1j),
          ]
source_R = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center, amplitude = 1),
          mp.Source(src, component=mp.Ey, size=source_size, center=source_center, amplitude = 1j),
          ]



#디자인 영역 minimum length, penalization parameter setting
minimum_length = 0.02  # minimum length scale (microns)
eta_i = 0.5  # blueprint (or intermediate) design field thresholding point (between 0 and 1)
eta_e = 0.55  # erosion design field thresholding point (between 0 and 1)
eta_d = 1 - eta_e  # dilation design field thresholding point (between 0 and 1)
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)
design_region_resolution = int(resolution)

# pixels of design region - using the design region resolution and size
Nx = int(round(design_region_width*design_region_resolution)) + 1
Ny = int(round(design_region_width*design_region_resolution)) + 1
Nz = 1

# setting the design region with design material and pixel information
design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny, Nz), Air, WS2_material, grid_type="U_MEAN")
design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(
        center=mp.Vector3(0, 0, 0),
        size=mp.Vector3(design_region_width, design_region_width, design_region_height),
    ),
)


# using conic_filter 
def mapping(x, eta, beta):
    # filter
    x = x.flatten()
    filtered_field = mpa.conic_filter(
        x,
        filter_radius,
        design_region_width,
        design_region_width,
        design_region_resolution,
        np.array([0,1]), #periodic condition
    )
    # output limit 0 to 1 (for binarization)
    projected_field = mpa.tanh_projection(filtered_field, beta, eta)

    projected_field = (
        projected_field + npa.rot90(projected_field,2)
    ) / 2  # C2 symmetry

    # projected_field = np.reshape(x,(Nx,Ny))
    mat = np.ones((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            if (i-(Nx-1)/2.0)**2 + (j-(Ny-1)/2.0)**2 > ((Nx-1)/2.0-2.01)**2:
                mat[i,j] = 0.0
            # if abs(i-(Nx-1)/2.0)> (Nx-1)/2.0-3 or abs(j-(Ny-1)/2.0)> (Ny-1)/2.0-3:
            #     mat[i,j] = 0.0
    projected_field = projected_field*mat
    projected_field = projected_field.flatten()
    return projected_field.flatten()



# creat block with same size with the design region
geometry = [
    mp.Block(
        center=mp.Vector3(0, 0, -0.25*(Lz+design_region_height)), size=mp.Vector3(Lx, Ly, 0.5*(Lz-design_region_height)), material=Glass
    ), # substrate
    mp.Block(
        center=design_region.center, size=design_region.size, material=design_variables
    ),
]




# %%
sim_L = mp.Simulation(
    cell_size=cell_size, 
    boundary_layers=[mp.PML(thickness = lpml, direction = mp.Z)],
    geometry=geometry,
    sources=source_L,
    # default_material=Air, # default empty
    resolution=resolution,
    k_point = mp.Vector3(0,0,0), # bloch boundary
    eps_averaging = False,
    force_complex_fields=False,
    extra_materials=[WS2_material, Air]
)
sim_R = mp.Simulation(
    cell_size=cell_size, 
    boundary_layers=[mp.PML(thickness = lpml, direction = mp.Z)],
    geometry=geometry,
    sources=source_R,
    # default_material=Air, # default empty
    resolution=resolution,
    k_point = mp.Vector3(0,0,0), # bloch boundary
    eps_averaging = False,
    force_complex_fields=False,
    extra_materials=[WS2_material, Air]

)


# Fourier transform result from FourierFields function in monitor (with monitor_position and monitor_size)
FF_size = Lx
center_pos =  0 #0.5/resolution
FourierFields_0_L = mpa.FourierFields(sim_L,mp.Volume(center=mp.Vector3(center_pos,center_pos,bot_monitor_pos),size=mp.Vector3(FF_size,FF_size,0)),mp.Ex)
FourierFields_1_L = mpa.FourierFields(sim_L,mp.Volume(center=mp.Vector3(center_pos,center_pos,bot_monitor_pos),size=mp.Vector3(FF_size,FF_size,0)),mp.Ey)

FourierFields_0_R = mpa.FourierFields(sim_R,mp.Volume(center=mp.Vector3(center_pos,center_pos,bot_monitor_pos),size=mp.Vector3(FF_size,FF_size,0)),mp.Ex)
FourierFields_1_R = mpa.FourierFields(sim_R,mp.Volume(center=mp.Vector3(center_pos,center_pos,bot_monitor_pos),size=mp.Vector3(FF_size,FF_size,0)),mp.Ey)


ob_list_L = [FourierFields_0_L, FourierFields_1_L]
ob_list_R = [FourierFields_0_R, FourierFields_1_R]
flux_LR_data, flux_LL_data, flux_RL_data, flux_RR_data = [], [], [], []
def J_L(fields_0, fields_1):
    flux_LR_data.append(npa.mean(0.5*npa.abs(fields_0[0,:,:]-1j*fields_1[0,:,:])**2))
    flux_LL_data.append(npa.mean(0.5*npa.abs(fields_0[0,:,:]+1j*fields_1[0,:,:])**2))
    return npa.mean(0.5*npa.abs(fields_0[0,:,:]-1j*fields_1[0,:,:])**2 - 0.5*npa.abs(fields_0[0,:,:]+1j*fields_1[0,:,:])**2)
def J_tot(fields_0, fields_1):
    flux_RL_data.append(npa.mean(0.5*npa.abs(fields_0[0,:,:]-1j*fields_1[0,:,:])**2))
    flux_RR_data.append(npa.mean(0.5*npa.abs(fields_0[0,:,:]+1j*fields_1[0,:,:])**2))
    return npa.mean(npa.abs(fields_0[0,:,:])**2+npa.abs(fields_1[0,:,:])**2)

opt_L = mpa.OptimizationProblem(
    simulation=sim_L,
    objective_functions=[J_L],
    objective_arguments=ob_list_L,
    design_regions=[design_region],
    fcen=frequency,
    df = 0,
    nf = 1,
    # decay_by=1e-1, # the field tolerance
    maximum_run_time = sim_time,
)

opt_R = mpa.OptimizationProblem(
    simulation=sim_R,
    objective_functions=[J_tot],
    objective_arguments=ob_list_R,
    design_regions=[design_region],
    fcen=frequency,
    df = 0,
    nf = 1,
    # decay_by=1e-1, # the field tolerance
    maximum_run_time = sim_time,
)

x0 = 0.5 * np.ones((Nx * Ny*Nz))
opt_L.update_design([x0]) # update the design as initial design
plt.figure()
opt_L.plot2D(
    True,
    output_plane=mp.Volume(center=(0, 0, 0), size=(Lx, 0, Lz)),
)
plt.savefig(folder_name+"/F_sim_map.jpg")
plt.close()

# %%
evaluation_history, evaluation_history_L, evaluation_history_R = [], [], []
flux_LR_array, flux_LL_array, flux_RR_array, flux_RL_array = [], [], [], [] 
cur_iter = [0]


def f(v, gradient, beta):
    print("Current iteration: {}".format(cur_iter[0] + 1))

    # f0_L, dJ_du_L = opt_L([v])  # compute objective and gradient
    # f0_R, dJ_du_R = opt_R([v])  # compute objective and gradient
    f0_L, dJ_du_L = opt_L([mapping(v, eta_i, beta)])  # compute objective and gradient
    f0_R, dJ_du_R = opt_R([mapping(v, eta_i, beta)])
    f0 = f0_L-f0_R
    dJ_du = dJ_du_L - dJ_du_R
    # Adjoint gradient
    if gradient.size > 0:
        gradient[:] = tensor_jacobian_product(mapping, 0)(
            v, eta_i, beta, np.squeeze(dJ_du)
        )  # backprop

    evaluation_history.append(np.real(f0))
    evaluation_history_L.append(np.real(f0_L))
    evaluation_history_R.append(np.real(f0_R))
    plt.figure()
    plt.imshow(np.rot90(np.reshape(design_variables.weights,[Nx, Ny])), cmap='binary')
    plt.colorbar()
    plt.savefig(folder_name+"/geometry_"+str(cur_iter[0]+1)+".jpg")
    plt.close()
    np.savetxt(folder_name+"/geometry_"+str(cur_iter[0]+1)+".txt",np.reshape(design_variables.weights,[Nx, Ny]))
    cur_iter[0] = cur_iter[0] + 1
    print("First FoM: {}".format(evaluation_history[0]))
    print("Current FoM: {}".format(np.real(f0)))
    np.savetxt(folder_name+"/FOM_history.txt", evaluation_history)
    np.savetxt(folder_name+"/FOM_history_L.txt", evaluation_history_L)
    np.savetxt(folder_name+"/FOM_history_R.txt", evaluation_history_R)
    plt.figure()
    plt.plot(evaluation_history, marker='o')
    plt.plot(evaluation_history_L, marker='o')
    plt.plot(evaluation_history_R, marker='o')
    plt.savefig(folder_name+"/FOM_history.jpg")
    plt.close()
    # with open(, "wb") as f:
    # print(np.real(flux_LR_data[0]))
    flux_LR_array.append(np.real(flux_LR_data[0+3*(cur_iter[0]-1)]))
    flux_LL_array.append(np.real(flux_LL_data[0+3*(cur_iter[0]-1)]))
    flux_RR_array.append(np.real(flux_RR_data[0+3*(cur_iter[0]-1)]))
    flux_RL_array.append(np.real(flux_RL_data[0+3*(cur_iter[0]-1)]))

    np.savetxt(folder_name+"/flux_LR.txt",flux_LR_array)
    np.savetxt(folder_name+"/flux_LL.txt",flux_LL_array)
    np.savetxt(folder_name+"/flux_RR.txt",flux_RR_array)
    np.savetxt(folder_name+"/flux_RL.txt",flux_RL_array)
    
    return np.real(f0)


algorithm = nlopt.LD_MMA # optimization algolithm
# MMA : asymptote shift

n = Nx * Ny * Nz  # number of parameters
# print("n is "+str(n))
# print(n)
# Initial guess - random initial point 
x = np.ones((Nx,Ny))

for i in range(Nx):
    for j in range(Ny):
        if (i-(Nx-1)/2)*(j-(Ny-1)/2) > 0 or abs(j-(Ny-1)/2)<5:
            x[i,j] = 0.6
        else:
            x[i,j] = 0.4
x = np.reshape(x,(n))


# lower and upper bounds (upper limit : 1, lower limit : 0)
lb = np.zeros((Nx * Ny* Nz,))
ub = np.ones((Nx * Ny* Nz,))

# Optimization parameter
cur_beta = 1
beta_scale = 2
num_betas = 8
update_factor = 15  # number of iterations between beta updates
ftol = 1e-5 

for iters in range(num_betas):
    solver = nlopt.opt(algorithm, n)
    solver.set_initial_step(0.01)
    solver.set_lower_bounds(lb) # lower bounds
    solver.set_upper_bounds(ub) # upper bounds
    solver.set_max_objective(lambda a, g: f(a, g, cur_beta))
    solver.set_maxeval(update_factor) # Set the maximum number of function evaluations
    solver.set_ftol_rel(ftol) # Set the relative tolerance for convergence
    x[:] = solver.optimize(x)
    cur_beta = cur_beta * beta_scale # Update the beta value for the next iteration
