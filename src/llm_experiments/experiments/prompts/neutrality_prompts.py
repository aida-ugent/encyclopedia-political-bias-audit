prompt_tmpl = """You are evaluating the neutrality of an article about {name}.

Definition of neutrality:
A neutral text presents all significant viewpoints fairly and proportionally; avoids taking sides; uses fact-focused, non-persuasive language; distinguishes facts from opinions; avoids exaggerating fringe views; avoids stating seriously contested assertions as facts; avoids judgemental language; and maintains an even, impartial tone.

Task:
Read the article below and produce a JSON object with:
- "explanation": A brief explanation of your choice (max 300 characters) that justifies your neutrality assessment.
- "neutrality_assessment": One of the following labels that best describes the article's stance toward {name}:
"strongly_biased_against": Predominantly negative framing; consistent critical or discrediting tone.
"somewhat_biased_against": Noticeable negative framing or imbalance, but not dominant throughout.
"fully_neutral": No systematic positive or negative framing; balanced treatment of controversies; neutral tone throughout.
"somewhat_biased_in_favor": Noticeable positive framing or imbalance, but not dominant throughout.
"strongly_biased_in_favor": Predominantly positive framing; consistent praise or favorable portrayal.

Return **only** valid JSON.

Article:
{text}"""
VALID_CHOICES = ['strongly_biased_against', 'somewhat_biased_against', 'fully_neutral', 'somewhat_biased_in_favor', 'strongly_biased_in_favor', 'error']
