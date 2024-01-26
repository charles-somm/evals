import yaml
from collections import Counter

from pathlib import Path
from random import Random
from jinja2 import Template
import evals
import evals.som_metrics

from evals.record import record_event
from evals.elsuite.sommelia.validation import validate_recommendation
from evals.som_metrics import get_accuracy
from evals.elsuite.modelgraded.classify_utils import (
    classify,
    sample_and_concat_n_completions,
)

from evals.elsuite.utils import PromptFn, scrub_formatting_from_prompt


class SommeliaEval(evals.Eval):
    """Evaluation function for wine recommendation based on a winelist and user criteria.
    Samples don't have a 'input' key, but a 'winelist_content' and 'criteria' key.
    Prompt is rendered from a template, with the winelist and criteria as arguments.

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
        # test_samples,
        template_id: str,
        temperature: float = 0.0,
        output_json: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # self.test_samples = test_samples
        self.template_id = template_id
        self.temperature = temperature
        self.output_json = output_json

    def _generate_samples(self):
        """Generate the samples input by rendering the prompt template
        for each combination of winelist and criteria and add it in the
        'input' key.
        Generating the samples dynamically allows to use the template_id as an
        argument, instead of having to create a new samples file for each template.
        """
        # Get the samples containing only the winelist and criteria
        samples = self.get_samples()

        # Get the prompt templates
        with open(Path(__file__).parent / "templates.yaml") as f:
            all_templates = yaml.safe_load(f)
        template = next(
            t for t in all_templates if t["template_id"] == self.template_id
        )

        # System prompt
        # TODO: Add examples as template parameters.
        system_prompt = template["system_template"]  # Doesn't change

        # Render the user prompt with each combination of winelist and criteria
        user_template = Template(template["user_template"])
        hydrated_samples = []
        for sample in samples:
            user_prompt = user_template.render(
                winelist=sample["winelist_content"], criteria=sample["criteria"]
            )
            sample["input"] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            hydrated_samples.append(sample)
        return hydrated_samples

    def run(self, recorder):
        samples = self._generate_samples()
        self.eval_all_samples(recorder, samples)

        # TODO: create custom metrics
        return get_accuracy(recorder.get_events("validation"))

    def eval_sample(self, test_sample: dict, args):
        completion_args = {
            "prompt": test_sample["input"],
            "temperature": self.temperature,
        }
        if self.output_json:
            completion_args["response_format"] = {"type": "json_object"}

        result = self.completion_fn(**completion_args)
        sampled = result.get_completions()[0]

        data = validate_recommendation(sampled, test_sample["listed_wines"])

        # Add winelist and criteria ids to the data
        data["winelist_id"] = test_sample["winelist_id"]
        data["criteria_id"] = test_sample["criteria_id"]
        data["temperature"] = self.temperature

        record_event(type="validation", data=data)


from evals.elsuite.modelgraded.classify import ModelBasedClassify


class SommeliaModelGradedEval(ModelBasedClassify):
    """Model-graded evaluation function for wine recommendation based on a winelist
    and user criteria.
    It inherits from the ModelBasedClassify class, which is used for the modelgraded
    evals.
    It expects a 'input' key containing the prompt and adds a 'completion' value.
    My samples don't have a 'input' key, but a 'winelist_content' and 'criteria' key.
    The eval_sample method is overriden to preprocess the sample and add the prompt
    as 'input'.
    Prompt is rendered from a template, with the winelist and criteria as arguments.

    Performs the following:
    - Renders the prompt as messages, with the samples criteria and winelist
    (and train samples)
    - Runs the prompt through the completion_fn
    - Runs the winelist, criteria and completion through the eval_completion_fn
    - Compute aggregated metrics
    - Record results
    """

    def __init__(
        self,
        samples_jsonl,
        template_id: str,
        temperature: float = 0.0,
        output_json: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.samples_jsonl = samples_jsonl
        self.template_id = template_id
        self.temperature = temperature
        self.output_json = output_json

    def _generate_samples(self):
        """Generate the samples input by rendering the prompt template
        for each combination of winelist and criteria and add it in the
        'input' key.
        Generating the samples dynamically allows to use the template_id as an
        argument, instead of having to create a new samples file for each template.
        """
        # Get the samples containing only the winelist and criteria
        samples = self.get_samples()

        # Get the prompt templates
        with open(Path(__file__).parent / "templates.yaml") as f:
            all_templates = yaml.safe_load(f)
        template = next(
            t for t in all_templates if t["template_id"] == self.template_id
        )

        # System prompt
        # TODO: Add examples as template parameters.
        system_prompt = template["system_template"]  # Doesn't change

        # Render the user prompt with each combination of winelist and criteria
        # (This could be done with the 'format_necessary' function instead of jinja2)
        user_template = Template(template["user_template"])
        hydrated_samples = []
        for sample in samples:
            user_prompt = user_template.render(
                winelist=sample["winelist_content"], criteria=sample["criteria"]
            )
            sample["input"] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            hydrated_samples.append(sample)
        return hydrated_samples

    def run(self, recorder):
        samples = self._generate_samples()

        self.eval_all_samples(recorder, samples)
        record_metrics = {}

        all_sample_metrics = recorder.get_metrics()
        if not all_sample_metrics:
            return record_metrics

        # record the counts
        choices = [m["choice"] for m in all_sample_metrics]
        counts = dict(Counter(choices))
        record_metrics.update({f"counts/{k}": v for k, v in counts.items()})

        # record the scores
        scores = [m["score"] for m in all_sample_metrics if m["score"] is not None]
        if scores:
            record_metrics["score"] = sum(scores) / len(scores)
        metascores = [m["metascore"] for m in all_sample_metrics if "metascore" in m]
        if metascores:
            record_metrics["metascore"] = sum(metascores) / len(metascores)

        return record_metrics

    def eval_sample(self, test_sample: dict, rng: Random) -> None:
        """Evaluate a single sample with a model-graded eval.
        'winelist' and 'criteria' values are passed to the 'classify' function
        as 'format_kwargs' beacause they must be added to the eval prompt dynamically.
        The prompt is rendered in the call to the PromptFn function class
        by 'format_necessary'.

        Recorded metrics are always: one of the self.choice_strings, or "__invalid__".
        """
        # process test_sample
        for k in self.mg.input_outputs:
            test_sample[k] = scrub_formatting_from_prompt(test_sample[k])

        # run policy completions
        completions = {}
        for k, v in self.mg.input_outputs.items():
            if v in test_sample:  # test_sample already has completion, skip.
                continue
            if self.multicomp_n > 1:
                completion = sample_and_concat_n_completions(
                    self.completion_fns,
                    prompt=test_sample[k],
                    template_i=self.mg.output_template,
                    sample_kwargs=self.sample_kwargs,
                    n=self.multicomp_n,
                )
            else:
                get_input_completion = PromptFn(
                    test_sample[k],
                    completion_fn=self.completion_fn,
                    # allow_missing=True,
                    **self.sample_kwargs,
                )
                completion, _ = get_input_completion()
            completions[v] = completion

        # run modelgraded eval
        metrics = {}
        choice, info = classify(
            mg=self.mg,
            completion_fn=self.eval_completion_fn,
            completion_kwargs=self.eval_kwargs,
            eval_type=self.eval_type,
            n=self.multicomp_n,
            match_fn=self.match_fn,
            format_kwargs={
                **completions,
                **test_sample,
                **self.modelgraded_spec_args,
                "allow_missing": True,
            },
        )
        metrics.update(dict(choice=choice, score=info["score"]))

        # run metaeval if requested
        if self.metaeval:
            assert "choice" in test_sample
            metrics["metascore"] = choice == test_sample["choice"]

        evals.record.record_metrics(**metrics)

        return choice
