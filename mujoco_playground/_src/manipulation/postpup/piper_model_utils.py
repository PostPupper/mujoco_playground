import jinja2
from etils import epath


def mjx_data():
    return {"mode": "mjx"}


def normal_data():
    return {"mode": "normal"}


def create_model_xml(
    template_xml: epath.Path, output_xml: epath.Path, mode: str
) -> str:
    template_xml = template_xml.read_text()
    jinja_template = jinja2.Template(template_xml)

    data_fn = {"mjx": mjx_data, "normal": normal_data}[mode]

    xml = jinja_template.render(data_fn())
    with open(output_xml, "w") as f:
        f.write(xml)
    return xml
