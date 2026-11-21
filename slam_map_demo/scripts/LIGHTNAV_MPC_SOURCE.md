# LightNav MPC source

`trajectory_mpc.py` is adapted from Light Origins' LightNav-0 `mujoco_demo/vln_mujoco/mpc.py`.
The copied source SHA-256 is `2de99fdf75b60c6836a645ae995c686417c93a5ccbc7df6d8b984d20c1d83ce1`.
It is licensed under Apache-2.0; see [LIGHTNAV_MPC_LICENSE](LIGHTNAV_MPC_LICENSE).

Local changes limit linear speed to 0.3 m/s and angular speed to 0.3 rad/s,
match the existing Go2 acceleration limits, bound IPOPT CPU time, accept
map-frame references, and discard results/errors from replaced generations.
The control service imports this local copy; it does not import the original checkout.
