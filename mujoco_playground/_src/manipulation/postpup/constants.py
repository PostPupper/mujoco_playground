from etils import epath

POSTPUP_DIR = epath.Path(__file__).parent
POSTPUP_XMLS = POSTPUP_DIR / "xmls"
POSTPUP_MESHES = POSTPUP_XMLS / "meshes"
POSTPUP_URDF = POSTPUP_XMLS / "urdf"

PLACE_LETTER_SCENE_XML = POSTPUP_XMLS / "place_letter_scene.xml"
PIPER_TEMPLATE_XML = POSTPUP_XMLS / "piper_template.xml"
PIPER_RENDERED_MJX_XML = POSTPUP_XMLS / "piper_rendered_mjx.xml"
PIPER_RENDERED_NORMAL_XML = POSTPUP_XMLS / "piper_rendered_normal.xml"
