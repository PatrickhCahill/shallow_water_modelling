# we are in ./scripts/run_model.py
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import ShallowWaterModel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "font.size": 15,        # default font size
    "axes.titlesize": 15,   # title font size
    "axes.labelsize": 15,   # x and y label font size
    "xtick.labelsize": 20,  # x tick labels
    "ytick.labelsize": 20,  # y tick labels
    "legend.fontsize": 12   # legend text
})

HIGH_RES_FIGURE_MUSCL_HYDROSTATIC = False


if __name__ == "__main__":
    T=1
    print("Running convergence tests for T =", T)
    print("BSTCS")
    Nxmax = 101
    Ntmax = 4001
    Nx_vals = [11, 21, 31, 46, Nxmax]
    Nt_vals = [int(Ntmax * Nx / Nxmax) for Nx in Nx_vals]

    for Nx, Nt in zip(Nx_vals, Nt_vals):
        dirname = f"conv_test_btcs_T{T}/Nx{Nx}_Nt{Nt}"
        model = ShallowWaterModel(
            method = "btcs",
            Nx = Nx,
            Nt = Nt,
            dirname = dirname,
            T=T
        )
        model.make_bottom_profile()
        model.create_solution_matrix()
        model.assign_initial_conditions()
        model.run()
        model.get_energies()
        model.get_masses()
        model.get_courant_numbers()
        model.save()
        model.plot_masses()
        model.plot_energies()
        model.plot_max_courant()
        model.height3d()
        model.velocity3d()
        model.heatmap_height()
        model.heatmap_velocity()
            # model.animate()

    errs_btcs = {}
    h_exact = pd.read_csv(f"./data/conv_test_btcs_T{T}/Nx{Nxmax}_Nt{Ntmax}/h.csv", index_col=0, dtype=np.float64).iloc[:,-1]
    hu_exact = pd.read_csv(f"./data/conv_test_btcs_T{T}/Nx{Nxmax}_Nt{Ntmax}/hu.csv", index_col=0, dtype=np.float64).iloc[:,-1]
    for Nx, Nt in zip(Nx_vals, Nt_vals):
        hu = pd.read_csv(f"./data/conv_test_btcs_T{T}/Nx{Nx}_Nt{Nt}/hu.csv", index_col=0, dtype=np.float64).iloc[:,-1]
        h = pd.read_csv(f"./data/conv_test_btcs_T{T}/Nx{Nx}_Nt{Nt}/h.csv", index_col=0, dtype=np.float64).iloc[:,-1]
        errs_btcs[Nx] = ((np.interp(hu.index.values, hu_exact.index.values, hu_exact.values) - hu.values)**2).sum() + ((np.interp(h.index.values, h_exact.index.values, h_exact.values) - h.values)**2).sum()

    print("MUSCL HYDROSTATIC")
    Nxmax = 81
    Ntmax = 4001
    Nx_vals = [11, 41, 51, 71, Nxmax]
    Nt_vals = [int(Ntmax * Nx / Nxmax) for Nx in Nx_vals]

    for Nx, Nt in zip(Nx_vals, Nt_vals):
        dirname = f"conv_test_muscl_hydrostatic_T{T}/Nx{Nx}_Nt{Nt}"
        model = ShallowWaterModel(
            method = "muscl_hydrostatic",
            Nx = Nx,
            Nt = Nt,
            dirname = dirname,
            T=T
        )
        model.make_bottom_profile()
        model.create_solution_matrix()
        model.assign_initial_conditions()
        model.run()
        model.get_energies()
        model.get_masses()
        model.get_courant_numbers()
        model.save()
        model.plot_masses()
        model.plot_energies()
        model.plot_max_courant()
        model.height3d()
        model.velocity3d()
        model.heatmap_height()
        model.heatmap_velocity()            # model.animate()

    errs_muscl_hydrostatic = {}
    h_exact = pd.read_csv(f"./data/conv_test_muscl_hydrostatic_T{T}/Nx{Nxmax}_Nt{Ntmax}/h.csv", index_col=0, dtype=np.float64).iloc[:,-1]
    hu_exact = pd.read_csv(f"./data/conv_test_muscl_hydrostatic_T{T}/Nx{Nxmax}_Nt{Ntmax}/hu.csv", index_col=0, dtype=np.float64).iloc[:,-1]
    for Nx, Nt in zip(Nx_vals, Nt_vals):
        hu = pd.read_csv(f"./data/conv_test_muscl_hydrostatic_T{T}/Nx{Nx}_Nt{Nt}/hu.csv", index_col=0, dtype=np.float64).iloc[:,-1]
        h = pd.read_csv(f"./data/conv_test_muscl_hydrostatic_T{T}/Nx{Nx}_Nt{Nt}/h.csv", index_col=0, dtype=np.float64).iloc[:,-1]
        errs_muscl_hydrostatic[Nx] = ((np.interp(hu.index.values, hu_exact.index.values, hu_exact.values) - hu.values)**2).sum() + ((np.interp(h.index.values, h_exact.index.values, h_exact.values) - h.values)**2).sum()

    print("LAX FRIEDRICHS")
    Nxmax = 101
    Ntmax = 201
    Nx_vals = [11, 41, 61, 81, Nxmax]
    Nt_vals = [int(Ntmax * Nx / Nxmax) for Nx in Nx_vals]

    for Nx, Nt in zip(Nx_vals, Nt_vals):
        dirname = f"conv_test_lax_T{T}/Nx{Nx}_Nt{Nt}"
        model = ShallowWaterModel(
            method = "lax_friedrichs",
            Nx = Nx,
            Nt = Nt,
            dirname = dirname,
            T=T
        )
        model.make_bottom_profile()
        model.create_solution_matrix()
        model.assign_initial_conditions()
        model.run()
        model.get_energies()
        model.get_masses()
        model.get_courant_numbers()
        model.save()
        model.plot_masses()
        model.plot_energies()
        model.plot_max_courant()
        model.height3d()
        model.velocity3d()
        model.heatmap_height()
        model.heatmap_velocity()            # model.animate()

    errs_lax = {}
    h_exact = pd.read_csv(f"./data/conv_test_lax_T{T}/Nx{Nxmax}_Nt{Ntmax}/h.csv", index_col=0, dtype=np.float64).iloc[:,-1]
    hu_exact = pd.read_csv(f"./data/conv_test_lax_T{T}/Nx{Nxmax}_Nt{Ntmax}/hu.csv", index_col=0, dtype=np.float64).iloc[:,-1]
    for Nx, Nt in zip(Nx_vals, Nt_vals):
        hu = pd.read_csv(f"./data/conv_test_lax_T{T}/Nx{Nx}_Nt{Nt}/hu.csv", index_col=0, dtype=np.float64).iloc[:,-1]
        h = pd.read_csv(f"./data/conv_test_lax_T{T}/Nx{Nx}_Nt{Nt}/h.csv", index_col=0, dtype=np.float64).iloc[:,-1]
        errs_lax[Nx] = ((np.interp(hu.index.values, hu_exact.index.values, hu_exact.values) - hu.values)**2).sum() + ((np.interp(h.index.values, h_exact.index.values, h_exact.values) - h.values)**2).sum()


    print("SYMPLECTIC")
    Nxmax = 101
    Ntmax = 10001
    Nx_vals = [11, 41, 81, Nxmax]
    Nt_vals = [int(Ntmax * Nx / Nxmax) for Nx in Nx_vals]

    for Nx, Nt in zip(Nx_vals, Nt_vals):
        dirname = f"conv_test_symplectic_T{T}/Nx{Nx}_Nt{Nt}"
        model = ShallowWaterModel(
            method = "symplectic",
            Nx = Nx,
            Nt = Nt,
            dirname = dirname,
            T=T
        )
        model.make_bottom_profile()
        model.create_solution_matrix()
        model.assign_initial_conditions()
        model.run()
        model.get_energies()
        model.get_masses()
        model.get_courant_numbers()
        model.save()
        model.plot_masses()
        model.plot_energies()
        model.plot_max_courant()
        model.height3d()
        model.velocity3d()
        model.heatmap_height()
        model.heatmap_velocity()            # model.animate()

    errs_symplectic = {}
    h_exact = pd.read_csv(f"./data/conv_test_symplectic_T{T}/Nx{Nxmax}_Nt{Ntmax}/h.csv", index_col=0, dtype=np.float64).iloc[:,-1]
    hu_exact = pd.read_csv(f"./data/conv_test_symplectic_T{T}/Nx{Nxmax}_Nt{Ntmax}/hu.csv", index_col=0, dtype=np.float64).iloc[:,-1]
    for Nx, Nt in zip(Nx_vals, Nt_vals):
        hu = pd.read_csv(f"./data/conv_test_symplectic_T{T}/Nx{Nx}_Nt{Nt}/hu.csv", index_col=0, dtype=np.float64).iloc[:,-1]
        h = pd.read_csv(f"./data/conv_test_symplectic_T{T}/Nx{Nx}_Nt{Nt}/h.csv", index_col=0, dtype=np.float64).iloc[:,-1]
        errs_symplectic[Nx] = ((np.interp(hu.index.values, hu_exact.index.values, hu_exact.values) - hu.values)**2).sum() + ((np.interp(h.index.values, h_exact.index.values, h_exact.values) - h.values)**2).sum()

    fig, ax = plt.subplots(figsize=(6,5), dpi=300, tight_layout=True)

        
    names = ["BTCS", "muscl hydrostatic", "lax", "symplectic"]
    colors = ["orange", "blue", "red", "purple" ]

    plt.rcParams.update({
        "font.size": 10,        # default font size
        "axes.titlesize": 10,   # title font size
        "axes.labelsize": 10,   # x and y label font size
        "xtick.labelsize": 10,  # x tick labels
        "ytick.labelsize": 10,  # y tick labels
        "legend.fontsize": 10   # legend text
    })

    for i, errs in enumerate([errs_btcs, errs_muscl_hydrostatic, errs_lax, errs_symplectic]):
        errs_x = [np.log(10/(i-1)) for i in list(errs.keys())[:-1]]
        errs_y = [np.log(errs[i]) for i in list(errs.keys())[:-1]]
        m = np.std(errs_y) / np.std(errs_x) * np.corrcoef(errs_x, errs_y)[0,1]
        b = np.mean(errs_y) - m * np.mean(errs_x)

        ax.plot(errs_x, errs_y, marker='o', label=f"{names[i]} Error. Order={m:.2f}", color=colors[i])

        # ax.plot(errs_x, [m*x + b for x in errs_x], label=f"Fit: slope={m:.2f}")

    ax.plot([-5, 0], [-5*1, 0], linestyle='--', color='black', label="First Order Reference")
    ax.plot([-3, 0], [-3*2, 0], linestyle='--', color='black', label="Second Order Reference")

    ax.set_xlabel(r"$\log(\Delta x)$")
    ax.set_ylabel(r"$\log(L^2$ error)")
    # ax.set_title("Convergence Test for Shallow Water Equations (FTCS)")
    ax.legend()
    plt.savefig(f"./data/convergence_test_shallow_water_equations_T{T}.png")

    plt.rcParams.update({
        "font.size": 15,        # default font size
        "axes.titlesize": 15,   # title font size
        "axes.labelsize": 15,   # x and y label font size
        "xtick.labelsize": 20,  # x tick labels
        "ytick.labelsize": 20,  # y tick labels
        "legend.fontsize": 12   # legend text
    })

    T=3
    print("Running convergence tests for T =", T)
    print("BTCS")
    Nxmax = 101
    Ntmax = 4000*3+1
    Nx_vals = [11, 21, 31, 46, Nxmax]
    Nt_vals = [int(Ntmax * Nx / Nxmax) for Nx in Nx_vals]

    for Nx, Nt in zip(Nx_vals, Nt_vals):
        dirname = f"conv_test_btcs_T{T}/Nx{Nx}_Nt{Nt}"
        model = ShallowWaterModel(
            method = "btcs",
            Nx = Nx,
            Nt = Nt,
            dirname = dirname,
            T=T
        )
        model.make_bottom_profile()
        model.create_solution_matrix()
        model.assign_initial_conditions()
        model.run()
        model.get_energies()
        model.get_masses()
        model.get_courant_numbers()
        model.save()
        model.plot_masses()
        model.plot_energies()
        model.plot_max_courant()
        model.height3d()
        model.velocity3d()
        model.heatmap_height()
        model.heatmap_velocity()            # model.animate()

    errs_btcs = {}
    h_exact = pd.read_csv(f"./data/conv_test_btcs_T{T}/Nx{Nxmax}_Nt{Ntmax}/h.csv", index_col=0, dtype=np.float64).iloc[:,-1]
    hu_exact = pd.read_csv(f"./data/conv_test_btcs_T{T}/Nx{Nxmax}_Nt{Ntmax}/hu.csv", index_col=0, dtype=np.float64).iloc[:,-1]
    for Nx, Nt in zip(Nx_vals, Nt_vals):
        hu = pd.read_csv(f"./data/conv_test_btcs_T{T}/Nx{Nx}_Nt{Nt}/hu.csv", index_col=0, dtype=np.float64).iloc[:,-1]
        h = pd.read_csv(f"./data/conv_test_btcs_T{T}/Nx{Nx}_Nt{Nt}/h.csv", index_col=0, dtype=np.float64).iloc[:,-1]
        errs_btcs[Nx] = ((np.interp(hu.index.values, hu_exact.index.values, hu_exact.values) - hu.values)**2).sum() + ((np.interp(h.index.values, h_exact.index.values, h_exact.values) - h.values)**2).sum()

    print("MUSCL HYDROSTATIC")
    Nxmax = 101
    Ntmax = 4000*3+1
    Nx_vals = [11, 41, 61, 81, Nxmax]
    Nt_vals = [int(Ntmax * Nx / Nxmax) for Nx in Nx_vals]

    for Nx, Nt in zip(Nx_vals, Nt_vals):
        dirname = f"conv_test_muscl_hydrostatic_T{T}/Nx{Nx}_Nt{Nt}"
        model = ShallowWaterModel(
            method = "muscl_hydrostatic",
            Nx = Nx,
            Nt = Nt,
            dirname = dirname,
            T=T
        )
        model.make_bottom_profile()
        model.create_solution_matrix()
        model.assign_initial_conditions()
        model.run()
        model.get_energies()
        model.get_masses()
        model.get_courant_numbers()
        model.save()
        model.plot_masses()
        model.plot_energies()
        model.plot_max_courant()
        model.height3d()
        model.velocity3d()
        model.heatmap_height()
        model.heatmap_velocity()            # model.animate()

    errs_muscl_hydrostatic = {}
    h_exact = pd.read_csv(f"./data/conv_test_muscl_hydrostatic_T{T}/Nx{Nxmax}_Nt{Ntmax}/h.csv", index_col=0, dtype=np.float64).iloc[:,-1]
    hu_exact = pd.read_csv(f"./data/conv_test_muscl_hydrostatic_T{T}/Nx{Nxmax}_Nt{Ntmax}/hu.csv", index_col=0, dtype=np.float64).iloc[:,-1]
    for Nx, Nt in zip(Nx_vals, Nt_vals):
        hu = pd.read_csv(f"./data/conv_test_muscl_hydrostatic_T{T}/Nx{Nx}_Nt{Nt}/hu.csv", index_col=0, dtype=np.float64).iloc[:,-1]
        h = pd.read_csv(f"./data/conv_test_muscl_hydrostatic_T{T}/Nx{Nx}_Nt{Nt}/h.csv", index_col=0, dtype=np.float64).iloc[:,-1]
        errs_muscl_hydrostatic[Nx] = ((np.interp(hu.index.values, hu_exact.index.values, hu_exact.values) - hu.values)**2).sum() + ((np.interp(h.index.values, h_exact.index.values, h_exact.values) - h.values)**2).sum()

    print("LAX FRIEDRICHS")
    Nxmax = 101
    Ntmax = 200*3+1
    Nx_vals = [11, 41, 61, 81, Nxmax]
    Nt_vals = [int(Ntmax * Nx / Nxmax) for Nx in Nx_vals]

    for Nx, Nt in zip(Nx_vals, Nt_vals):
        dirname = f"conv_test_lax_T{T}/Nx{Nx}_Nt{Nt}"
        model = ShallowWaterModel(
            method = "lax_friedrichs",
            Nx = Nx,
            Nt = Nt,
            dirname = dirname,
            T=T
        )
        model.make_bottom_profile()
        model.create_solution_matrix()
        model.assign_initial_conditions()
        model.run()
        model.get_energies()
        model.get_masses()
        model.get_courant_numbers()
        model.save()
        model.plot_masses()
        model.plot_energies()
        model.plot_max_courant()
        model.height3d()
        model.velocity3d()
        model.heatmap_height()
        model.heatmap_velocity()            # model.animate()

    errs_lax = {}
    h_exact = pd.read_csv(f"./data/conv_test_lax_T{T}/Nx{Nxmax}_Nt{Ntmax}/h.csv", index_col=0, dtype=np.float64).iloc[:,-1]
    hu_exact = pd.read_csv(f"./data/conv_test_lax_T{T}/Nx{Nxmax}_Nt{Ntmax}/hu.csv", index_col=0, dtype=np.float64).iloc[:,-1]
    for Nx, Nt in zip(Nx_vals, Nt_vals):
        hu = pd.read_csv(f"./data/conv_test_lax_T{T}/Nx{Nx}_Nt{Nt}/hu.csv", index_col=0, dtype=np.float64).iloc[:,-1]
        h = pd.read_csv(f"./data/conv_test_lax_T{T}/Nx{Nx}_Nt{Nt}/h.csv", index_col=0, dtype=np.float64).iloc[:,-1]
        errs_lax[Nx] = ((np.interp(hu.index.values, hu_exact.index.values, hu_exact.values) - hu.values)**2).sum() + ((np.interp(h.index.values, h_exact.index.values, h_exact.values) - h.values)**2).sum()


    print("SYMPLECTIC")
    Nxmax = 101
    Ntmax = 10000*3+1
    Nx_vals = [11, 41, 81, Nxmax]
    Nt_vals = [int(Ntmax * Nx / Nxmax) for Nx in Nx_vals]

    for Nx, Nt in zip(Nx_vals, Nt_vals):
        dirname = f"conv_test_symplectic_T{T}/Nx{Nx}_Nt{Nt}"
        model = ShallowWaterModel(
            method = "symplectic",
            Nx = Nx,
            Nt = Nt,
            dirname = dirname,
            T=T
        )
        model.make_bottom_profile()
        model.create_solution_matrix()
        model.assign_initial_conditions()
        model.run()
        model.get_energies()
        model.get_masses()
        model.get_courant_numbers()
        model.save()
        model.plot_masses()
        model.plot_energies()
        model.plot_max_courant()
        model.height3d()
        model.velocity3d()
        model.heatmap_height()
        model.heatmap_velocity()
            # model.animate()

    errs_symplectic = {}
    h_exact = pd.read_csv(f"./data/conv_test_symplectic_T{T}/Nx{Nxmax}_Nt{Ntmax}/h.csv", index_col=0, dtype=np.float64).iloc[:,-1]
    hu_exact = pd.read_csv(f"./data/conv_test_symplectic_T{T}/Nx{Nxmax}_Nt{Ntmax}/hu.csv", index_col=0, dtype=np.float64).iloc[:,-1]
    for Nx, Nt in zip(Nx_vals, Nt_vals):
        hu = pd.read_csv(f"./data/conv_test_symplectic_T{T}/Nx{Nx}_Nt{Nt}/hu.csv", index_col=0, dtype=np.float64).iloc[:,-1]
        h = pd.read_csv(f"./data/conv_test_symplectic_T{T}/Nx{Nx}_Nt{Nt}/h.csv", index_col=0, dtype=np.float64).iloc[:,-1]
        errs_symplectic[Nx] = ((np.interp(hu.index.values, hu_exact.index.values, hu_exact.values) - hu.values)**2).sum() + ((np.interp(h.index.values, h_exact.index.values, h_exact.values) - h.values)**2).sum()

    fig, ax = plt.subplots(figsize=(6,5), dpi=300, tight_layout=True)

        
    names = ["BTCS", "muscl hydrostatic", "lax", "symplectic"]
    colors = ["orange", "blue", "red", "purple" ]
    
    plt.rcParams.update({
        "font.size": 10,        # default font size
        "axes.titlesize": 10,   # title font size
        "axes.labelsize": 10,   # x and y label font size
        "xtick.labelsize": 10,  # x tick labels
        "ytick.labelsize": 10,  # y tick labels
        "legend.fontsize": 10   # legend text
    })



    for i, errs in enumerate([errs_btcs, errs_muscl_hydrostatic, errs_lax, errs_symplectic]):
        errs_x = [np.log(10/(i-1)) for i in list(errs.keys())[:-1]]
        errs_y = [np.log(errs[i]) for i in list(errs.keys())[:-1]]
        m = np.std(errs_y) / np.std(errs_x) * np.corrcoef(errs_x, errs_y)[0,1]
        b = np.mean(errs_y) - m * np.mean(errs_x)

        ax.plot(errs_x, errs_y, marker='o', label=f"{names[i]} Error. Order={m:.2f}", color=colors[i])

        # ax.plot(errs_x, [m*x + b for x in errs_x], label=f"Fit: slope={m:.2f}")

    ax.plot([-5, 0], [-5*1, 0], linestyle='--', color='black', label="First Order Reference")
    ax.plot([-3, 0], [-3*2, 0], linestyle='--', color='black', label="Second Order Reference")

    ax.set_xlabel(r"$\log(\Delta x)$")
    ax.set_ylabel(r"$\log(L^2$ error)")
    # ax.set_title("Convergence Test for Shallow Water Equations (FTCS)")
    ax.legend()
    plt.savefig(f"./data/convergence_test_shallow_water_equations_T{T}.png")

    plt.rcParams.update({
        "font.size": 15,        # default font size
        "axes.titlesize": 15,   # title font size
        "axes.labelsize": 15,   # x and y label font size
        "xtick.labelsize": 20,  # x tick labels
        "ytick.labelsize": 20,  # y tick labels
        "legend.fontsize": 12   # legend text
    })



    x = np.linspace(-5,5,201)
    a = np.cos(np.pi/2 * (x+5)/5)-2  # Wavy bottom
    h0 = 0 - a[:] + 2/5 * np.exp(-x**2)
    u0 = np.zeros(201)
    u0[100:] = 4
    h_u_override = h0 * u0
    model = ShallowWaterModel(dirname="shallow_water_lax_shock", method="lax_friedrichs", Nt=2001, T=1.0, Nx=201)
    model.make_bottom_profile(wave=False)
    model.create_solution_matrix()
    model.assign_initial_conditions(h_u_override=h_u_override)
    model.run()
    model.get_energies()
    model.get_masses()
    model.get_courant_numbers()
    model.save()
    model.plot_initial_conditions()
    model.plot_masses()
    model.plot_energies()
    model.plot_max_courant()
    model.height3d()
    model.velocity3d()
    model.heatmap_height()
    model.heatmap_velocity()  
    


    model = ShallowWaterModel(dirname="shallow_water_muscl_shock", method="muscl_hydrostatic", Nt=2001, T=1.0, Nx=201)
    model.make_bottom_profile(wave=False)
    model.create_solution_matrix()
    model.assign_initial_conditions(h_u_override=h_u_override)
    model.run()
    model.get_energies()
    model.get_masses()
    model.get_courant_numbers()
    model.save()
    model.plot_initial_conditions()
    model.plot_masses()
    model.plot_energies()
    model.plot_max_courant()
    model.height3d()
    model.velocity3d()
    model.heatmap_height()
    model.heatmap_velocity()  
    


    a = np.zeros(201) - 1
    a[30:] = -2
    model = ShallowWaterModel(dirname="shallow_water_lax_discontinuous", method="lax_friedrichs", Nt=2001, T=2.0, Nx=201)
    model.make_bottom_profile(override=a)
    model.create_solution_matrix()
    model.assign_initial_conditions()
    model.run()
    model.get_energies()
    model.get_masses()
    model.get_courant_numbers()
    model.save()
    model.plot_initial_conditions()
    model.plot_masses()
    model.plot_energies()
    model.plot_max_courant()
    model.height3d()
    model.velocity3d()
    model.heatmap_height()
    model.heatmap_velocity()      

    model = ShallowWaterModel(dirname="shallow_water_muscl_discontinuous", method="muscl_hydrostatic", Nt=2001, T=2.0, Nx=201)
    model.make_bottom_profile(override=a)
    model.create_solution_matrix()
    model.assign_initial_conditions()
    model.run()
    model.get_energies()
    model.get_masses()
    model.get_courant_numbers()
    model.save()
    model.plot_initial_conditions()
    model.plot_masses()
    model.plot_energies()
    model.plot_max_courant()
    model.height3d()
    model.velocity3d()
    model.heatmap_height()
    model.heatmap_velocity()      

    if HIGH_RES_FIGURE_MUSCL_HYDROSTATIC:
        model = ShallowWaterModel(dirname="kurganov_petrova", method="muscl_hydrostatic", Nt=10001, T=3, Nx=501)
        model.make_bottom_profile()
        model.create_solution_matrix()
        model.assign_initial_conditions()
        model.run()
        model.get_energies()
        model.get_masses()
        model.get_courant_numbers()
        model.save()
        model.plot_masses()
        model.plot_energies()
        model.plot_max_courant()
        model.height3d()
        model.velocity3d()
        model.heatmap_height()
        model.heatmap_velocity()
        model.animate(show=False)
