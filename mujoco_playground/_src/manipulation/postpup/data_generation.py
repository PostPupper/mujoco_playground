from typing import Tuple, Generator
from mujoco_playground._src.manipulation.postpup import model_utils, constants
import mujoco
import numpy as np

from etils import epath
import click
import tqdm


def generate_fk(model, data, joint_angles: np.ndarray, site_name: str) -> np.ndarray:
    data.qpos[:] = joint_angles
    mujoco.mj_kinematics(model, data)
    site = data.site(site_name)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, site.xmat)
    return np.concatenate((site.xpos, quat))


def get_joint_limits(model) -> np.ndarray:
    joint_limits = np.zeros((model.nq, 2))
    for i in range(model.nq):
        joint_limits[i, 0] = model.jnt_range[i, 0]
        joint_limits[i, 1] = model.jnt_range[i, 1]
    return joint_limits


def generate_data(model, data, N: int, site_name) -> np.ndarray:
    joint_limits = get_joint_limits(model)
    joint_angles = np.random.uniform(
        joint_limits[:, 0], joint_limits[:, 1], (N, model.nq)
    )

    ik_data = np.zeros((N, model.nq + 7))
    for i in tqdm.tqdm(range(N)):
        pos_quat = generate_fk(model, data, joint_angles[i], site_name)
        ik_data[i, :] = np.concatenate((joint_angles[i], pos_quat))
    return ik_data


def write_data(
    model_xml: epath.Path, output_file: epath.Path, N: int, site_name: str
) -> None:
    model, data = model_utils.load_callback(xml_path=model_xml)
    ik_data = generate_data(model, data, N, site_name)
    np.savez(
        output_file,
        data=ik_data,
        nq=model.nq,
        model_xml=model_xml.as_posix(),
        site_name=site_name,
    )


@click.command()
@click.option("--model", type=click.Choice(["piper"]), required=True)
@click.option("--output_file", type=epath.Path, required=True)
@click.option("--num_poses", type=int, required=True)
@click.option("--site_name", type=str, default="gripper_site_x_forward", required=False)
def main(model: str, output_file: epath.Path, num_poses: int, site_name: str):
    if model == "piper":
        model_xml = constants.PIPER_RENDERED_NORMAL_XML
    write_data(model_xml, output_file, N=num_poses, site_name=site_name)


if __name__ == "__main__":
    main()
