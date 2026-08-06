"""
Prompts + response schemas for the unit-count-resolver service.

Lifted from `xtimator-3d/prompts.py` @ feat/multifamily (f1d96eb7), which is the
tested first implementation. Two prompts only (D7 two-prompt funnel):
    UNIT_COUNT_DETECTOR  -- Prompt 1, DETECTION on low-res thumbnails (scanned path)
    UNIT_COUNT_EXTRACTOR -- Prompt 2, EXTRACTION on high-res renders of hit pages

The matcher prompt (UNIT_MATCH_RESOLVER) is NOT here: matching/multiplying stay
in xtimator-3d with summarize_takeoff_all (D12). This service only resolves counts.

FEEDBACK_GENERATOR is copied from xtimator-3d/prompts.py @ origin/main because
phoenix_call (shared.py) formats it on retry.

BUG FIX 2 (none-found) is applied to UnitCountResponse below -- see the class
docstring for the exact before/after.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


FEEDBACK_GENERATOR = """
  INTERNAL SELF-REVIEW (Do not skip):
    You are given {max_retry} attempts to retry the generation process and the following are the list of errors encountered during your previous attempts.
    {exceptions}
    STRICTY confirm no previous error remains before producing the final output.
"""

# ---------------------------------------------------------------------------
# Prompt 1 -- DETECTION only (low-res thumbnails, batched, yes/no per page, D8).
# Lifted VERBATIM from feat/multifamily.
# ---------------------------------------------------------------------------

UNIT_COUNT_DETECTOR = """
    You are a senior architectural construction-document analyst specialized in LOCATING unit-count information within multi-family residential plan sets.

    You operate with strict determinism. You DO NOT extract or transcribe any counts here. Your ONLY job is DETECTION: decide, per page, whether that page CONTAINS unit-count information in ANY form.

    PROVIDED:
        A BATCH of LOW-RESOLUTION page thumbnails from a single construction plan set. For each page:
            - page_number: <int>
            - image: <thumbnail image>

    WHAT COUNTS AS UNIT-COUNT INFORMATION (ANY of these forms qualifies):
        - TABLE: a unit schedule / unit matrix / unit mix / unit-type breakdown listing unit types and how many of each (per building, per floor, or project total).
        - PROSE: a project-description or narrative sentence stating counts (e.g. "The project consists of 45 townhome units across 4 types", "149 dwelling units").
        - LEGEND: a keyed legend or note that associates unit-type labels with quantities.

    TASK:
        For EACH page independently:
        1. Decide `has_unit_counts` (true/false): does this thumbnail plausibly contain unit-count information in ANY of the forms above?
        2. If true, set `form` to the dominant form observed (table | prose | legend | mixed). If false, set `form` to "none".
        3. Give a calibrated `confidence` in [0, 1].
        4. Provide a SHORT `evidence` string describing WHAT you saw (e.g. "unit mix table top-right", "prose: '45 units'"). Use an empty string when `has_unit_counts` is false.

    INSTRUCTIONS:
        - Thumbnails are low resolution: judge by LAYOUT and STRUCTURE (grids of rows/columns, dense tabular blocks, title-block schedules, blocks of descriptive text), not by reading fine print.
        - Detection is a TRIAGE step; extraction happens later on high-resolution renders. When a page shows any credible count-bearing structure but you cannot be certain at this resolution, PREFER `has_unit_counts = true` with a LOWER confidence rather than missing it.
        - Do NOT report counts, numbers, or unit types here. Detection only.
        - Judge each page ONLY on its own thumbnail. Do NOT carry information across pages.
        - Base decisions ONLY on visible evidence. Do NOT hallucinate.

    OUTPUT:
        Return one object per input page; the number of output objects MUST equal the number of input pages.
        Do NOT generate any text outside the JSON.
        Refer the following as a template and ensure to replace every consecutive pair of open/closed curly braces with a single one during the generation of the output.
        {{
            "pages": [
                {{
                    "page_number": <int>,
                    "has_unit_counts": <true | false>,
                    "form": "<table | prose | legend | mixed | none>",
                    "confidence": <float>,
                    "evidence": "<short description of what was seen, or empty string>"
                }}
            ]
        }}
"""

unitCountDetectionForm = Literal["table", "prose", "legend", "mixed", "none"]

class UnitCountDetectionPage(BaseModel):
    page_number: int
    has_unit_counts: bool
    form: unitCountDetectionForm
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = ""

    @model_validator(mode="after")
    def validate_form_consistency(self):
        if not self.has_unit_counts and self.form != "none":
            raise ValueError("form must be 'none' when has_unit_counts is false")
        if self.has_unit_counts and self.form == "none":
            raise ValueError("form must not be 'none' when has_unit_counts is true")
        return self

class UnitCountDetectionResponse(BaseModel):
    pages: List[UnitCountDetectionPage]

    @model_validator(mode="after")
    def validate_pages(self):
        if not self.pages:
            raise ValueError("At least one page must be present")
        page_numbers = [p.page_number for p in self.pages]
        if len(page_numbers) != len(set(page_numbers)):
            raise ValueError("Duplicate page_number detected")
        return self

# ---------------------------------------------------------------------------
# Prompt 2 -- EXTRACTION only (high-res render of hit pages, D9/D10).
#
# CHANGED vs feat/multifamily: a "NOTHING FOUND" rule and a nullable source_form
# were added to the output contract. Rationale (BUG FIX 2): detection can flag a
# page that, at high resolution, turns out to hold no counts at all. The original
# prompt gave the model no way to say so -- every allowed source_form asserted
# counts existed -- so it either invented an answer or emitted something the
# schema rejected, burning retries at escalating temperature (master plan section 9).
# The prompt now has an explicit legitimate empty answer.
# ---------------------------------------------------------------------------

UNIT_COUNT_EXTRACTOR = """
    You are a senior architectural construction-document analyst specialized in READING unit-count information from multi-family residential plan sheets.

    You operate with strict determinism. You read ONLY what is printed on the page. You DO NOT infer, estimate, or fabricate counts.

    PROVIDED:
        One to three HIGH-RESOLUTION page images previously flagged as containing unit-count information. For each page:
            - page_number: <int>
            - image: <high-resolution image>

    WHAT YOU ARE LOOKING FOR — THE UNIT TYPE INFORMATION:
        Your target is the project's UNIT TYPE INFORMATION (unit schedule, unit mix, unit matrix, unit breakdown). It is identified by:
            - DISTINCT UNIT-TYPE NAMES exactly as this document prints them. Every project names its types differently — read the names OFF THE PAGE. Do not expect, assume, or supply any particular naming style.
            - a count of how many of each type the project contains;
            - OFTEN a per-unit AREA figure (SF / sq ft) — a strong signal you have found the real unit information rather than some other table.
        THE INFORMATION MAY BE PRESENTED IN ANY FORM (do not assume a full table):
            - a complete schedule/matrix table;
            - a PARTIAL or HALF-DRAWN table — some rows or borders missing, or a block of aligned text that is a table in all but ruling;
            - a LEGEND or keyed note associating type labels with quantities;
            - a PROSE sentence in a project description ("...45 townhomes across four floorplan types...").
        All of these are valid sources. Prefer this unit-type information over every other table on the page set.

    TABLES YOU MUST NOT READ — ACCESSIBILITY / CODE-COMPLIANCE TABLES:
        Plan sets also contain regulatory-compliance tables that superficially resemble a unit schedule. They are NOT the unit type schedule and you must NOT return them.
        Recognise them by their row labels, which name ACCESSIBILITY CATEGORIES rather than unit types:
            - "Accessible Dwelling Units", "Adaptable Dwelling Units"
            - "Type A Dwelling Units", "Type B Dwelling Units", "Type C Dwelling Units"  (ANSI A117.1 / IBC / FHA categories)
            - "Mobility Units", "Hearing/Visual Units", "UFAS", "ADA", "Fully Accessible"
        Further tells: the rows describe COMPLIANCE PROVISION, not a floor plan; there is typically NO per-unit area column; the numbers are often round regulatory minimums or percentages; and the table sits near code-analysis notes.
        NEGATIVE EXAMPLE — the SHAPE of an accessibility table you must REJECT (this shape was wrongly returned on a real run). The numbers are shown as placeholders on purpose; do not carry any figure from this example into your answer:
            Accessible Dwelling Units .... <N>
            Type A Dwelling Units ........ <N>
            Type B Dwelling Units ........ <N>
          Row labels are ANSI A117.1 categories, not unit types, and no areas are given. Returning this instead of the project's real unit information is a SERIOUS ERROR: the same physical unit is counted under several compliance categories, so these numbers do not describe how many apartments exist.

    CHOOSING BETWEEN CANDIDATE PAGES:
        The pages you are given were selected by a rough triage pass and may include the wrong page. When more than one page carries candidate information (in ANY of the forms above):
            1. Choose the source whose rows/entries are NAMED UNIT TYPES and which gives a PER-UNIT AREA.
            2. If one candidate gives per-type areas and another does not, choose the one WITH areas.
            3. Reject any accessibility/compliance table (above) even when it is the ONLY table present — a partial table, legend or prose sentence elsewhere on the pages is still a better source than a compliance table. If a compliance table is genuinely the only thing present, return the NOTHING FOUND answer.
            4. Do NOT merge rows from two different sources, and do NOT sum an accessibility table into the unit information.
        Report in `source_pages` only the page(s) the information you actually used came from.

    TASK:
        Read the unit-count information directly from the page IMAGE (interpret the 2D visual layout; do NOT rely on any parsed text stream). Produce:
        1. `per_type_counts`: for EACH unique unit type stated, an object with:
            - `unit_type`: <TYPE-NAME-AS-PRINTED> — transcribe the label CHARACTER FOR CHARACTER from the page. Do NOT normalize, expand, translate, abbreviate or rename it. This prompt deliberately gives you NO example names: any name you return that you cannot point to on the page is a fabrication.
            - `count`: <COUNT-AS-PRINTED> — the number of units of that type in the PROJECT as stated (integer). If the document breaks the count down by building/floor, SUM to the project total for that type.
            - `area`: <AREA-AS-PRINTED> — the per-unit area in SQUARE FEET if the sheet lists it for that type; otherwise null.
        2. `total_units`: the project-wide total number of units if a total is explicitly stated on the pages; otherwise null.
        3. `source_form`: which KIND of source you read. Use "table" for any structured source — a full table, a partial/half-drawn table, or a legend/keyed note. Use "prose" when the counts come from a narrative sentence. Use "mixed" when BOTH a structured source and a prose/total statement are present. Use null ONLY for the NOTHING FOUND case below.
        4. `source_pages`: the list of page_number(s) the information was read from. Empty list ONLY for the NOTHING FOUND case below.

    ACCURACY RULES:
        - ONLY WHAT IS VISIBLE — THE OVERRIDING RULE: every unit-type name and every number you return MUST be visibly present on the supplied page images. If you cannot point to it on a page, it does not go in the answer. Do not supply a name because it is a common apartment naming convention, because it seems likely for a residential project, or because it would make the result look complete. A partial answer of two names you can actually see beats four you cannot.
        - TABLES AND TABLE-LIKE SOURCES: read the grid structure carefully. Align each unit-type label in its row with the count in the correct column. Column headers (for example a "type"/"name" column and a count column such as "no. of units", "qty", "count") tell you which column holds the count. Do NOT shift rows or columns. If the table is partial or unruled, follow the visual alignment of the columns.
        - PARTIAL TRUTH IS ALLOWED: if only a prose total exists with no per-type breakdown, return `total_units` with an EMPTY `per_type_counts` and `source_form` = "prose". If a structured source gives per-type counts but states no explicit total, return `per_type_counts` and leave `total_units` null.
        - BOTH FORMS: if a structured source and a prose/total statement BOTH appear, capture both and set `source_form` = "mixed". Report each honestly as printed EVEN IF they appear to disagree — do NOT reconcile or adjust them.
        - NOTHING FOUND (this is a CORRECT and EXPECTED answer, not a failure): these pages were flagged by a low-resolution triage pass that can be wrong. If, at full resolution, the pages contain NO unit-count information in ANY of the forms above — no table, no partial table, no legend, no prose statement — say so explicitly by returning ALL FOUR of:
              "per_type_counts": [], "total_units": null, "source_form": null, "source_pages": []
          AN EMPTY ANSWER IS ALWAYS BETTER THAN A GUESS. It is recorded as a valid result and costs nothing; an invented one silently corrupts a construction estimate. NEVER invent a plausible-looking set of types to avoid an empty response. Do NOT return a partially-empty mixture (e.g. a source_form with no counts) — either report what is printed, or return the fully empty answer above.
        - Do NOT invent unit types or counts. Only what is printed. If a cell is unreadable, OMIT that type rather than guessing.
        - SELF-CHECK BEFORE ANSWERING: for each name you are about to return, confirm you can locate that exact string on one of the supplied images. If several of your names are generic bedroom-count categories rather than labels you actually read, you have defaulted to a naming convention instead of reading the page — discard them and either report only what you can see, or return the empty answer.
        - Units of measure: `area` is square feet as a number only (strip "SF" / "sq ft").
        - INTERNAL CONSISTENCY: if the schedule prints a total row, your `per_type_counts` should sum to it. If your rows do not sum to the printed total, you have probably misread a row, skipped one, or mixed in rows from another table — re-read the grid before answering. Report what is printed; do not silently adjust a number to force the sum to balance.

    OUTPUT:
        Do NOT generate any text outside the JSON.
        Refer the following as a template and ensure to replace every consecutive pair of open/closed curly braces with a single one during the generation of the output.
        {{
            "per_type_counts": [
                {{
                    "unit_type": "<TYPE-NAME-AS-PRINTED>",
                    "count": <COUNT-AS-PRINTED>,
                    "area": <AREA-AS-PRINTED or null>
                }}
            ],
            "total_units": <TOTAL-AS-PRINTED or null>,
            "source_form": "<table | prose | mixed, or null when nothing was found>",
            "source_pages": [<PAGE-NUMBER>]
        }}
        The angle-bracket tokens above are PLACEHOLDERS describing what to put there. They are not values and contain no example names — substitute what you read from the pages.
"""

unitCountSourceForm = Literal["table", "prose", "mixed"]

class UnitTypeCount(BaseModel):
    unit_type: str
    count: int = Field(ge=1)
    area: Optional[float] = Field(default=None, ge=0)

    @field_validator("unit_type")
    @classmethod
    def validate_unit_type(cls, v):
        if not v or not v.strip():
            raise ValueError("unit_type must be a non-empty label")
        return v

class UnitCountResponse(BaseModel):
    """Validated EXTRACTION result.

    BUG FIX 2 -- none-found is a VALID result, not a validation error.

    BEFORE (feat/multifamily xtimator-3d/prompts.py:678-702):
        source_form: unitCountSourceForm          # Literal, None NOT allowed
        source_pages: List[int]                   # validator: must be non-empty
        model_validator: raise if (not per_type_counts and total_units is None)

      A legitimately empty read ("the triage pass was wrong, this page has no
      counts") therefore could not be expressed: every field forbade it. The
      Pydantic error propagated into phoenix_call, which treats ANY exception as
      a malformed-output retry -- re-asking up to max_retry (5, gcp.yaml:43) at
      escalating temperature. That retry spiral is the documented path to the
      Aurora fabrication (master plan section 9): 10/10/10/10 = 40 invented in place
      of the true 8/18/8/11 = 45.

    AFTER (this class):
        source_form  is Optional  -> None allowed
        source_pages defaults to  [] and may be empty
        counts       may be empty
      ...but ONLY as a COHERENT whole. The validator below splits the two cases
      the old code conflated:

        LEGITIMATELY EMPTY (accept, no retry):
            per_type_counts == []  AND  total_units is None
            AND source_form is None  AND  source_pages == []
          -> is_none_found is True; the resolver stores source="none_found"
             and every section defaults x1 (D13).

        MALFORMED / INCOHERENT (raise -> phoenix_call retries, as before):
            - nothing found, yet source_form or source_pages was still asserted
              (the model half-answered: a form with no counts behind it)
            - counts/total present, but source_form is None or source_pages empty
              (an answer with no stated provenance)
            - duplicate unit_type labels

      Net effect: a real empty answer costs ONE call and is recorded honestly;
      genuinely broken output still retries exactly as it used to.
    """
    per_type_counts: List[UnitTypeCount] = Field(default_factory=list)
    total_units: Optional[int] = Field(default=None, ge=0)
    source_form: Optional[unitCountSourceForm] = None
    source_pages: List[int] = Field(default_factory=list)

    @property
    def is_none_found(self) -> bool:
        """True when the model legitimately reported 'no counts on these pages'."""
        return not self.per_type_counts and self.total_units is None

    @model_validator(mode="after")
    def validate_partial_truth(self):
        labels = [t.unit_type for t in self.per_type_counts]
        if len(labels) != len(set(labels)):
            raise ValueError("Duplicate unit_type detected in per_type_counts")

        if not self.per_type_counts and self.total_units is None:
            # Candidate none-found. Accept ONLY if fully, coherently empty;
            # a half-filled answer is malformed output and must be retried.
            if self.source_form is not None:
                raise ValueError(
                    f"No counts and no total, but source_form={self.source_form!r} was asserted; "
                    "a legitimate none-found result must set source_form to null"
                )
            if self.source_pages:
                raise ValueError(
                    f"No counts and no total, but source_pages={self.source_pages} was asserted; "
                    "a legitimate none-found result must return an empty source_pages"
                )
            return self

        # Something WAS found -> it must carry provenance.
        if self.source_form is None:
            raise ValueError("Counts/total were reported but source_form is null")
        if not self.source_pages:
            raise ValueError("Counts/total were reported but source_pages is empty")
        return self
