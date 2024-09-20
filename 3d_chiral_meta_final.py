# %%
import meep as mp
import meep.adjoint as mpa
import numpy as np
# from autograd import numpy as npa
# from autograd import tensor_jacobian_product, grad
# import nlopt
from matplotlib import pyplot as plt
# from matplotlib.patches import Circle
import h5py

# %%
def def_WS2():

    # default unit length is 1 μm
    um_scale = 1.0

    # conversion factor for eV to 1/μm [=1/hc]
    eV_um_scale = um_scale / 1.23984193

    # ------------------------------------------------------------------
    # WS2 bulk from  https://doi.org/10.1021/acs.nanolett.3c02051
    # wavelength range: 0.6 - 0.8 μm

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

# WS2_material = mp.Medium(index=4.2)
Glass = mp.Medium(index=1.5)

Air = mp.Medium(epsilon_diag=mp.Vector3(1,1,1))

WS2_material = def_WS2()

# setting the simulation space
design_region_height = 0.25
design_region_width = 0.38
lpml = 0.5
Lx = design_region_width
Ly = design_region_width
Lz = 2*lpml + design_region_height + 2.0
cell_size = mp.Vector3(Lx, Ly, Lz)
resolution = 100
# wavelength, frequency setting
wavelength = 0.680
frequency = 1/wavelength
width = 0.5
n_freq= 100

sim_time = 20
folder_name = "geometry_380_digit"
binarization_on = True
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



#Design space, minimum length, penalization factor setting
minimum_length = 0.02  # minimum length scale (microns)
eta_i = 0.5  # blueprint (or intermediate) design field thresholding point (between 0 and 1)
eta_e = 0.55  # erosion design field thresholding point (between 0 and 1)
eta_d = 1 - eta_e  # dilation design field thresholding point (between 0 and 1)
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)
design_region_resolution = int(resolution)

# number of pixels of the design space in x and y direction - determined by resolution and design region size
Nx = int(round(design_region_width*design_region_resolution)) + 1
Ny = int(round(design_region_width*design_region_resolution)) + 1
Nz = 1

# design region setting by the design space and the materials
design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny, Nz), Air, WS2_material, grid_type="U_MEAN")
design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(
        center=mp.Vector3(0, 0, 0),
        size=mp.Vector3(design_region_width, design_region_width, design_region_height),
    ),
)

# generating design space block
geometry = [
    mp.Block(
        center=mp.Vector3(0, 0, -0.25*(Lz+design_region_height)), size=mp.Vector3(Lx, Ly, 0.5*(Lz-design_region_height)), material=Glass
    ),
    mp.Block(
        center=design_region.center, size=design_region.size, material=design_variables
    ),
]


# %%
design_variables.weights = np.loadtxt(folder_name+"/geometry_120.txt")
if binarization_on:
    design_variables.weights = np.round(design_variables.weights)


sim_L_empty = mp.Simulation(
    cell_size=cell_size, 
    boundary_layers=[mp.PML(thickness = lpml, direction = mp.Z)],
    # geometry=geometry,
    sources=source_L,
    # default_material=Air, # Hollow space
    resolution=resolution,
    k_point = mp.Vector3(0,0,0), # bloch boundary
    eps_averaging = False,
    # force_complex_fields=True,
    # extra_materials=[WS2_material, Air],
)

flux_top_empty_L = sim_L_empty.add_flux(frequency,frequency*width*0.5,n_freq,mp.FluxRegion(center=mp.Vector3(0,0,top_monitor_pos), size=mp.Vector3(Lx,Ly,0)))
flux_bot_empty_L = sim_L_empty.add_flux(frequency,frequency*width*0.5,n_freq,mp.FluxRegion(center=mp.Vector3(0,0,bot_monitor_pos), size=mp.Vector3(Lx,Ly,0)))

dft_fields_top_empty_L = sim_L_empty.add_dft_fields([mp.Ex, mp.Ey, mp.Hx, mp.Hy],frequency,frequency*width*0.5,n_freq, center=mp.Vector3(0,0,top_monitor_pos), size=mp.Vector3(Lx, Ly,0))
dft_fields_bot_empty_L = sim_L_empty.add_dft_fields([mp.Ex, mp.Ey, mp.Hx, mp.Hy],frequency,frequency*width*0.5,n_freq, center=mp.Vector3(0,0,bot_monitor_pos), size=mp.Vector3(Lx, Ly,0))


sim_L_empty.run(until_after_sources=sim_time)

sim_L_empty.output_dft(dft_fields_top_empty_L, folder_name+"/sim_L_empty_top")
sim_L_empty.output_dft(dft_fields_bot_empty_L, folder_name+"/sim_L_empty_bot")


empty_flux_top_array_L = np.array(mp.get_fluxes(flux_top_empty_L))
empty_flux_bot_array_L = np.array(mp.get_fluxes(flux_bot_empty_L))
empty_flux_top_data_L = sim_L_empty.get_flux_data(flux_top_empty_L)
freq_array_L = np.array(mp.get_flux_freqs(flux_top_empty_L))

sim_L = mp.Simulation(
    cell_size=cell_size, 
    boundary_layers=[mp.PML(thickness = lpml, direction = mp.Z)],
    geometry=geometry,
    sources=source_L,
    default_material=Air, # hollow space
    resolution=resolution,
    k_point = mp.Vector3(0,0,0), # bloch boundary
    eps_averaging = False,
    # force_complex_fields=True,
    extra_materials=[WS2_material, Air],
)


flux_top_L = sim_L.add_flux(frequency,frequency*width*0.5,n_freq,mp.FluxRegion(center=mp.Vector3(0,0,top_monitor_pos), size=mp.Vector3(Lx,Ly,0)))
sim_L.load_minus_flux_data(flux_top_L, empty_flux_top_data_L)

flux_bot_L = sim_L.add_flux(frequency,frequency*width*0.5,n_freq,mp.FluxRegion(center=mp.Vector3(0,0,bot_monitor_pos), size=mp.Vector3(Lx,Ly,0)))
dft_fields_top = sim_L.add_dft_fields([mp.Ex, mp.Ey, mp.Hx, mp.Hy],frequency,frequency*width*0.5,n_freq, center=mp.Vector3(0,0,top_monitor_pos), size=mp.Vector3(Lx, Ly,0))
dft_fields_bot = sim_L.add_dft_fields([mp.Ex, mp.Ey, mp.Hx, mp.Hy],frequency,frequency*width*0.5,n_freq, center=mp.Vector3(0,0,bot_monitor_pos), size=mp.Vector3(Lx, Ly,0))

plt.figure()
sim_L.plot2D(
    output_plane=mp.Volume(center=(0, 0, 0), size=(Lx, 0, Lz)),
)
plt.savefig(folder_name+"/sim_map_xz.jpg")
plt.close()
plt.figure()
sim_L.plot2D(
    output_plane=mp.Volume(center=(0, 0, 0), size=(Lx, Ly, 0)),
)
plt.savefig(folder_name+"/sim_map_xy.jpg")
plt.close()



sim_L.run(until_after_sources=sim_time)
sim_L.output_dft(dft_fields_top, folder_name+"/sim_L_top")
sim_L.output_dft(dft_fields_bot, folder_name+"/sim_L_bot")


flux_top_array_L = np.array(mp.get_fluxes(flux_top_L))
flux_bot_array_L = np.array(mp.get_fluxes(flux_bot_L))
freq_array_L = np.array(mp.get_flux_freqs(flux_top_empty_L))



plt.figure()
plt.plot(1/freq_array_L, flux_bot_array_L/empty_flux_bot_array_L)
plt.plot(1/freq_array_L, -flux_top_array_L/empty_flux_top_array_L)
plt.plot(1/freq_array_L, 1-flux_bot_array_L/empty_flux_bot_array_L+flux_top_array_L/empty_flux_bot_array_L)
plt.savefig(folder_name+"/flux_L_sim.jpg")
plt.close()
np.savetxt(folder_name+"/flux_L_sim.txt", (freq_array_L,flux_bot_array_L/empty_flux_bot_array_L,flux_top_array_L/empty_flux_top_array_L,
                              1-flux_bot_array_L/empty_flux_bot_array_L+flux_top_array_L/empty_flux_bot_array_L))

# %%
sim_R_empty = mp.Simulation(
    cell_size=cell_size, 
    boundary_layers=[mp.PML(thickness = lpml, direction = mp.Z)],
    # geometry=geometry,
    sources=source_R,
    # default_material=Air, # hollow space
    resolution=resolution,
    k_point = mp.Vector3(0,0,0), # bloch boundary
    eps_averaging = False,
    # force_complex_fields=True,
    # extra_materials=[WS2_material, Air],
)

flux_top_empty_R = sim_R_empty.add_flux(frequency,frequency*width*0.5,n_freq,mp.FluxRegion(center=mp.Vector3(0,0,top_monitor_pos), size=mp.Vector3(Lx,Ly,0)))

flux_bot_empty_R = sim_R_empty.add_flux(frequency,frequency*width*0.5,n_freq,mp.FluxRegion(center=mp.Vector3(0,0,bot_monitor_pos), size=mp.Vector3(Lx,Ly,0)))


dft_fields_top_empty_R = sim_R_empty.add_dft_fields([mp.Ex, mp.Ey, mp.Hx, mp.Hy],frequency,frequency*width*0.5,n_freq, center=mp.Vector3(0,0,top_monitor_pos), size=mp.Vector3(Lx, Ly,0))
dft_fields_bot_empty_R = sim_R_empty.add_dft_fields([mp.Ex, mp.Ey, mp.Hx, mp.Hy],frequency,frequency*width*0.5,n_freq, center=mp.Vector3(0,0,bot_monitor_pos), size=mp.Vector3(Lx, Ly,0))

sim_R_empty.run(until_after_sources=sim_time)

sim_R_empty.output_dft(dft_fields_top_empty_R, folder_name+"/sim_R_empty_top")
sim_R_empty.output_dft(dft_fields_bot_empty_R, folder_name+"/sim_R_empty_bot")


empty_flux_top_array_R = np.array(mp.get_fluxes(flux_top_empty_R))
empty_flux_bot_array_R = np.array(mp.get_fluxes(flux_bot_empty_R))
empty_flux_top_data_R = sim_R_empty.get_flux_data(flux_top_empty_R)
freq_array_R = np.array(mp.get_flux_freqs(flux_top_empty_R))


# %%
sim_R = mp.Simulation(
    cell_size=cell_size, 
    boundary_layers=[mp.PML(thickness = lpml, direction = mp.Z)],
    geometry=geometry,
    sources=source_R,
    default_material=Air, # hollow space
    resolution=resolution,
    k_point = mp.Vector3(0,0,0), # bloch boundary
    eps_averaging = False,
    # force_complex_fields=True,
    extra_materials=[WS2_material, Air],
)

flux_top_R = sim_R.add_flux(frequency,frequency*width*0.5,n_freq,mp.FluxRegion(center=mp.Vector3(0,0,top_monitor_pos), size=mp.Vector3(Lx,Ly,0)))
sim_R.load_minus_flux_data(flux_top_R, empty_flux_top_data_R)

flux_bot_R = sim_R.add_flux(frequency,frequency*width*0.5,n_freq,mp.FluxRegion(center=mp.Vector3(0,0,bot_monitor_pos), size=mp.Vector3(Lx,Ly,0)))

dft_fields_top = sim_R.add_dft_fields([mp.Ex, mp.Ey, mp.Hx, mp.Hy],frequency,frequency*width*0.5,n_freq, center=mp.Vector3(0,0,top_monitor_pos), size=mp.Vector3(Lx, Ly,0))
dft_fields_bot = sim_R.add_dft_fields([mp.Ex, mp.Ey, mp.Hx, mp.Hy],frequency,frequency*width*0.5,n_freq, center=mp.Vector3(0,0,bot_monitor_pos), size=mp.Vector3(Lx, Ly,0))

sim_R.run(until_after_sources=sim_time)
sim_R.output_dft(dft_fields_top, folder_name+"/sim_R_top")
sim_R.output_dft(dft_fields_bot, folder_name+"/sim_R_bot")

flux_top_array_R = np.array(mp.get_fluxes(flux_top_R))
flux_bot_array_R = np.array(mp.get_fluxes(flux_bot_R))
freq_array_R = np.array(mp.get_flux_freqs(flux_top_empty_R))

# %%
plt.figure()
plt.plot(1/freq_array_R, flux_bot_array_R/empty_flux_bot_array_R)
plt.plot(1/freq_array_R, -flux_top_array_R/empty_flux_top_array_R)
plt.plot(1/freq_array_R, 1-flux_bot_array_R/empty_flux_bot_array_R+flux_top_array_R/empty_flux_bot_array_R)
plt.savefig(folder_name+"/flux_R_sim.jpg")
plt.close()
# %%
plt.figure()
plt.plot(1/freq_array_R, flux_bot_array_L/empty_flux_bot_array_L)
plt.plot(1/freq_array_R, flux_bot_array_R/empty_flux_bot_array_R)
plt.savefig(folder_name+"/RL_compare_flux.jpg")
plt.close()

np.savetxt(folder_name+"/flux_R_sim.txt", (1/freq_array_R,flux_bot_array_R/empty_flux_bot_array_R,-flux_top_array_R/empty_flux_top_array_R,
                              1-flux_bot_array_R/empty_flux_bot_array_R+flux_top_array_R/empty_flux_bot_array_R))


def h5_load_fields (FileName1, EorH, freq_num, is_forward_field: bool = True, coo=0):
    # output: target field = adjoint field^* <- in this case, you should set "is_forward_field = False"
    if EorH == 'E':
        R_coo=['ex_'+str(freq_num)+'.r','ey_'+str(freq_num)+'.r','ez_'+str(freq_num)+'.r']
        I_coo=['ex_'+str(freq_num)+'.i','ey_'+str(freq_num)+'.i','ez_'+str(freq_num)+'.i']
    if EorH == 'H':
        R_coo=['hx_'+str(freq_num)+'.r','hy_'+str(freq_num)+'.r','hz_'+str(freq_num)+'.r']
        I_coo=['hx_'+str(freq_num)+'.i','hy_'+str(freq_num)+'.i','hz_'+str(freq_num)+'.i']
    if EorH != 'E'and EorH !='H':
        print('enter ''E'' or ''H''')
        return None
        
    # Load DFT Field
    # print(FileName1)
    hf = h5py.File(FileName1, 'r')
    A=hf.get(R_coo[coo])
    R=np.array(A) # Real value

    B=hf.get(I_coo[coo])
    I=np.array(B)*1j # Imaginary value

    Field=R+I # Complex field

    
    if is_forward_field:
        return Field

    else:
        return np.conjugate(Field)


L_RCP_flux, L_LCP_flux, LCP_flux_empty = [], [], []
fcen, width, n_freq = 1/0.68, 0.25, 100
freq_array = np.linspace(fcen-0.5*fcen*width, fcen+0.5*fcen*width, n_freq)
for i in range(n_freq):
    L_RCP_flux.append(
    np.sum(-np.real((h5_load_fields(folder_name+"/sim_L_bot.h5",'E',i,False,0) + 1j*h5_load_fields(folder_name+"/sim_L_bot.h5",'E',i,False,1))*
    (h5_load_fields(folder_name+"/sim_L_bot.h5",'H',i,True,1) + 1j*h5_load_fields(folder_name+"/sim_L_bot.h5",'H',i,True,0)))))
    L_LCP_flux.append(
    np.sum(-np.real((h5_load_fields(folder_name+"/sim_L_bot.h5",'E',i,False,0) - 1j*h5_load_fields(folder_name+"/sim_L_bot.h5",'E',i,False,1))*
    (h5_load_fields(folder_name+"/sim_L_bot.h5",'H',i,True,1) - 1j*h5_load_fields(folder_name+"/sim_L_bot.h5",'H',i,True,0)))))
    LCP_flux_empty.append(
    np.sum(-np.real((h5_load_fields(folder_name+"/sim_L_empty_top.h5",'E',i,False,0) - 1j*h5_load_fields(folder_name+"/sim_L_empty_top.h5",'E',i,False,1))*
    (h5_load_fields(folder_name+"/sim_L_empty_top.h5",'H',i,True,1) - 1j*h5_load_fields(folder_name+"/sim_L_empty_top.h5",'H',i,True,0)))))

# %%
R_RCP_flux, R_LCP_flux, RCP_flux_empty = [], [], []
fcen, width, n_freq = 1/0.68, 0.25, 100
freq_array = np.linspace(fcen-0.5*fcen*width, fcen+0.5*fcen*width, n_freq)
for i in range(n_freq):
    R_RCP_flux.append(
    np.sum(-np.real((h5_load_fields(folder_name+"/sim_R_bot.h5",'E',i,False,0) + 1j*h5_load_fields(folder_name+"/sim_R_bot.h5",'E',i,False,1))*
    (h5_load_fields(folder_name+"/sim_R_bot.h5",'H',i,True,1) + 1j*h5_load_fields(folder_name+"/sim_R_bot.h5",'H',i,True,0)))))
    R_LCP_flux.append(
    np.sum(-np.real((h5_load_fields(folder_name+"/sim_R_bot.h5",'E',i,False,0) - 1j*h5_load_fields(folder_name+"/sim_R_bot.h5",'E',i,False,1))*
    (h5_load_fields(folder_name+"/sim_R_bot.h5",'H',i,True,1) - 1j*h5_load_fields(folder_name+"/sim_R_bot.h5",'H',i,True,0)))))
    RCP_flux_empty.append(
    np.sum(-np.real((h5_load_fields(folder_name+"/sim_R_empty_top.h5",'E',i,False,0) + 1j*h5_load_fields(folder_name+"/sim_R_empty_top.h5",'E',i,False,1))*
    (h5_load_fields(folder_name+"/sim_R_empty_top.h5",'H',i,True,1) + 1j*h5_load_fields(folder_name+"/sim_R_empty_top.h5",'H',i,True,0)))))

# %%
plt.figure()
plt.plot(1/freq_array,np.array(L_LCP_flux)/np.array(LCP_flux_empty))
plt.plot(1/freq_array,np.array(L_RCP_flux)/np.array(LCP_flux_empty))
plt.plot(1/freq_array,np.array(R_LCP_flux)/np.array(RCP_flux_empty))
plt.plot(1/freq_array,np.array(R_RCP_flux)/np.array(RCP_flux_empty))
plt.legend(['T_LL', 'T_RL', 'T_LR', 'T_RR'])
plt.savefig(folder_name+"/CP_T.jpg")
plt.close()
np.savetxt(folder_name+"/CP_T.txt", (1/freq_array, 
                                    np.array(L_LCP_flux)/np.array(LCP_flux_empty), 
                                    np.array(L_RCP_flux)/np.array(LCP_flux_empty), 
                                    np.array(R_LCP_flux)/np.array(RCP_flux_empty), 
                                    np.array(R_RCP_flux)/np.array(RCP_flux_empty),
                                    ))
