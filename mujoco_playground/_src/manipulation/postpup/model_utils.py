import jinja2
from etils import epath
from mujoco_playground._src.manipulation.postpup import constants
import mujoco
from mujoco import viewer
from typing import Callable, Dict, Optional

from mujoco_playground._src import mjx_env


def mjx_data():
    return {"mode": "mjx"}


def normal_data():
    return {"mode": "normal"}


DATA_FNS = {"mjx": mjx_data, "normal": normal_data}
XML_PATHS = {
    "mjx": constants.PIPER_RENDERED_MJX_XML,
    "normal": constants.PIPER_RENDERED_NORMAL_XML,
}


def get_assets() -> Dict[str, bytes]:
    assets = {}
    mjx_env.update_assets(assets, constants.POSTPUP_XMLS, "*.xml")
    mjx_env.update_assets(assets, constants.POSTPUP_MESHES)
    return assets


def create_model_xml(
    template_xml: epath.Path, output_xml: epath.Path, mode: str
) -> str:
    template_xml = template_xml.read_text()
    jinja_template = jinja2.Template(template_xml)

    xml = jinja_template.render(DATA_FNS[mode]())
    with open(output_xml, "w") as f:
        f.write(xml)
    return xml


def create_all_model_variations():
    for mode in XML_PATHS:
        create_model_xml(constants.PIPER_TEMPLATE_XML, XML_PATHS[mode], mode)


def load_callback(
    xml_path: epath.Path, model=None, data=None, control_fn: Optional[Callable] = None
):

    create_all_model_variations()

    model = mujoco.MjModel.from_xml_path(
        xml_path.as_posix(),
        assets=get_assets(),
    )
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)

    ctrl_dt = 0.02
    sim_dt = 0.004
    n_substeps = int(round(ctrl_dt / sim_dt))
    model.opt.timestep = sim_dt

    if control_fn is not None:
        mujoco.set_mjcb_control(control_fn)

    return model, data


def visualize_model(xml_path: epath.Path, control_fn: Optional[Callable] = None):
    viewer.launch(loader=lambda: load_callback(xml_path, control_fn=control_fn))
