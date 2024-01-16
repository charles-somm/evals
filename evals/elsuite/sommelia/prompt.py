import yaml

from pathlib import Path

from jinja2 import Template


def render_template(template_id: str, winelist: str, criteria: str) -> list[dict]:
    """Renders a template with the given winelist and criteria.
    TODO: Add examples separately.
    """
    cwd = Path(__file__).parent
    with open(cwd / "templates.yaml") as f:
        templates = yaml.safe_load(f)
    template = next(t for t in templates if t["template_id"] == template_id)
    system_prompt = template["system_template"]
    user_template = Template(template["user_template"])
    user_prompt = user_template.render(winelist=winelist, criteria=criteria)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return messages
