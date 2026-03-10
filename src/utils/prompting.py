import json
import pandas as pd


def build_candidates_block(candidates_df: pd.DataFrame) -> str:
    entries = []

    for _, row in candidates_df.iterrows():
        entries.append(
            {
                "code": row["code"],
                "title": row["title"],
                "description": row["text"],
            }
        )

    return json.dumps(entries, ensure_ascii=False, indent=2)


def generate_classification_prompt(
    title: str,
    abstract: str,
    candidates_df: pd.DataFrame,
    max_secondary: int = 3,
) -> str:
    title = str(title).strip() if title is not None else ""
    abstract = str(abstract).strip() if abstract is not None else ""

    candidates_block = build_candidates_block(candidates_df)

    prompt = f"""
You are an expert in economic activity classification.

Your task is to classify the following patent into the most relevant NACE Rev. 2.1 class.
Use the patent title and abstract to infer the economic activity most closely related to the invention.

Focus on the economic activity that would most likely develop, produce, commercialize, or implement the patented invention.
Do not classify the patent only by its scientific topic or by generic references to technology.

Important:
- Choose the primary code only from the candidate classes provided below.
- The primary code should represent the single most plausible principal economic activity associated with the patented invention.
- Secondary codes are optional.
- Secondary codes must also be selected only from the candidate classes provided below.
- Secondary codes should be included only if they represent genuinely relevant alternative or complementary economic activities.
- Do not invent codes.
- Return only valid candidate codes.

Patent title:
\"\"\"{title}\"\"\"

Patent abstract:
\"\"\"{abstract}\"\"\"

Candidate NACE classes:
{candidates_block}

Return ONLY a JSON object with the following structure:
{{
  "primary_code": "...",
  "secondary_codes": ["...", "..."]
}}

Rules:
- "primary_code" must contain exactly one code.
- "secondary_codes" can contain up to {max_secondary} codes.
- Do not include the primary code inside secondary_codes.
- If no secondary code is appropriate, return an empty list.
- Do not add explanations.
"""
    return prompt.strip()
