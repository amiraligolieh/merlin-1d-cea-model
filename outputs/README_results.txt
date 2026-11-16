Results generated using the installed NASA CEA backend.
Three phi values: lean 0.85, stoichiometric 1.0, rich = OF_st/2.36.
RP-1 empirical formula C1H1.95, so OF_st is an approximation, not a measured composition.
Model assumes fixed pressure and nozzle dimensions; mdot from Pc*At/cstar (CEA).
Thrust = mdot*g0*Isp_vac - P_ambient*Ae.
No flow separation, losses, throttling, transient operation or off-design shock modelling.
CEA finite area chamber: iac=False, ac_at=Ac/At; frozen throat n_frz=3.
Species plots: X is mole fraction (dimensionless), NOT kinetic production rate.
Species mass flow = mdot*Y_species, and net change is outlet minus chamber mass flux.
Actual Merlin geometrical parameters in source code were assumed rather than sourced.
The black sea-level 845 kN marker is SpaceX published reference, not a modeled datum.
SpaceX: https://www.spacex.com/assets/media/falcon-users-guide-2025-05-09.pdf
NASA CEA API: https://nasa.github.io/cea/interfaces/python_api.html
NASA CEA FAC: https://nasa.github.io/cea/examples/rocket/example9.html
NASA CEA Frozen: https://nasa.github.io/cea/examples/rocket/example12.html
