import evals
import evals.sommelia_metrics

from evals.record import record_event
from evals.elsuite.sommelia.prompt import render_template
from evals.elsuite.sommelia.validation import validate_recommendation
from evals.sommelia_metrics import get_accuracy


class SommeliaEval(evals.Eval):
    """Evaluation function for wine recommendation based on a winelist and user criteria.
    Performs the following:
    - Renders the prompt as messages, with the samples criteria and winelist
    (and train samples)
    - Runs the prompt through the completion_fn
    - Evaluate the response
    - Compute aggregated metrics
    - Record results
    """

    def __init__(
        self,
        test_samples,
        template_id: str,
        temperature: float = 0.0,
        output_json: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.test_samples = test_samples
        self.template_id = template_id
        self.temperature = temperature
        self.output_json = output_json

    def run(self, recorder):
        test_samples = evals.get_jsonl(self.test_samples)
        self.eval_all_samples(recorder, test_samples)

        # TODO: create custom metrics
        return get_accuracy(recorder.get_events("validation"))
        # return {"accuracy": 42.0}

    def eval_sample(self, test_sample, args):
        prompt = render_template(
            template_id=self.template_id,
            winelist=test_sample["winelist_content"],
            criteria=test_sample["criteria"],
        )

        completion_args = {
            "prompt": prompt,
            "temperature": self.temperature,
        }
        if self.output_json:
            completion_args["response_format"] = {"type": "json_object"}

        result = self.completion_fn(**completion_args)
        sampled = result.get_completions()[0]

        data = validate_recommendation(sampled, test_sample["listed_wines"])

        record_event(type="validation", data=data)
