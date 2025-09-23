import json
from django.http import JsonResponse
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.core.files.uploadedfile import UploadedFile
from django.conf import settings
import google.generativeai as genai
from PyPDF2 import PdfReader
from docx import Document
from rest_framework import status
import logging

# Setup logging
logger = logging.getLogger(__name__)

# -------------------- Utility Functions --------------------

def _ensure_list(value):
    """Ensure a value is always returned as a list"""
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]

def normalize_result(result: dict) -> dict:
    """
    Normalize AI output to consistent schema
    {
      "tor_text": "string",
      "criteria": [...],
      "candidates": [...],
      "comparison_matrix": [...],
      "final_recommendation": { ... }
    }
    """
    default_scores = {
        "general_qualifications": {
            "education": 0.0,
            "years_of_experience": 0.0,
            "total": 0.0
        },
        "adequacy_for_assignment": {
            "relevant_project_experience": 0.0,
            "donor_experience": 0.0,
            "regional_experience": 0.0,
            "total": 0.0
        },
        "specific_skills_competencies": {
            "technical_skills": 0.0,
            "language_proficiency": 0.0,
            "certifications": 0.0,
            "total": 0.0
        },
        "total_score": 0.0
    }

    default_candidate = {
        "candidate_name": "Unnamed Candidate",
        "recommendation": "Not Evaluated",
        "scores": default_scores,
        "summary_justification": {
            "key_strengths": "None provided.",
            "key_weaknesses": "None provided."
        },
        "detailed_evaluation": []
    }

    default_result = {
        "tor_text": "",
        "criteria": [
            {"criterion": "Education", "weight": 10},
            {"criterion": "Years of Experience", "weight": 10},
            {"criterion": "Relevant Project Experience", "weight": 25},
            {"criterion": "Donor Experience (WB, ADB, etc.)", "weight": 15},
            {"criterion": "Regional Experience", "weight": 10},
            {"criterion": "Technical Skills", "weight": 15},
            {"criterion": "Language Proficiency", "weight": 10},
            {"criterion": "Certifications", "weight": 5}
        ],
        "candidates": [],
        "comparison_matrix": [],
        "final_recommendation": {
            "best_candidate": "None",
            "final_decision": "None Suitable",
            "justification": "No suitable candidates found."
        }
    }

    if not isinstance(result, dict):
        logger.warning("AI output is not a dict: %s", result)
        return default_result

    # Normalize candidates
    candidates = _ensure_list(result.get("candidates", []))
    normalized_candidates = []
    for candidate in candidates:
        normalized_candidate = default_candidate.copy()
        normalized_candidate.update({
            "candidate_name": candidate.get("candidate_name", "Unnamed Candidate"),
            "recommendation": candidate.get("recommendation", "Not Evaluated"),
            "scores": {**default_scores, **candidate.get("scores", {})},
            "summary_justification": {
                "key_strengths": candidate.get("summary_justification", {}).get("key_strengths", "None provided."),
                "key_weaknesses": candidate.get("summary_justification", {}).get("key_weaknesses", "None provided.")
            },
            "detailed_evaluation": _ensure_list(candidate.get("detailed_evaluation", []))
        })
        normalized_candidates.append(normalized_candidate)

    # Normalize comparison matrix
    comparison_matrix = _ensure_list(result.get("comparison_matrix", []))
    normalized_matrix = []
    for item in comparison_matrix:
        normalized_matrix.append({
            "candidate_name": item.get("candidate_name", "Unnamed Candidate"),
            "total_score": item.get("total_score", 0.0),
            "rank": item.get("rank", 0)
        })

    # Normalize final recommendation
    final_recommendation = result.get("final_recommendation", {})
    normalized_final_recommendation = {
        "best_candidate": final_recommendation.get("best_candidate", "None"),
        "final_decision": final_recommendation.get("final_decision", "None Suitable"),
        "justification": final_recommendation.get("justification", "No suitable candidates found.")
    }

    normalized_result = {
        "tor_text": result.get("tor_text", ""),
        "criteria": _ensure_list(result.get("criteria", default_result["criteria"])),
        "candidates": normalized_candidates,
        "comparison_matrix": normalized_matrix,
        "final_recommendation": normalized_final_recommendation
    }

    logger.debug("Normalized result: %s", normalized_result)
    return normalized_result

# -------------------- Main Class View --------------------

@method_decorator(csrf_exempt, name='dispatch')
class CompareCVsView(View):
    def post(self, request):
        tor = request.POST.get('tor')

        # Collect files from request
        cv_files = []
        if hasattr(request, 'FILES') and request.FILES:
            for field_name in request.FILES.keys():
                cv_files.extend(request.FILES.getlist(field_name))

        # Validation
        if not tor:
            return JsonResponse({'error': 'ToR is required'}, status=status.HTTP_400_BAD_REQUEST)
        if not cv_files:
            return JsonResponse({'error': 'At least one CV is required'}, status=status.HTTP_400_BAD_REQUEST)
        if len(cv_files) > 10:
            return JsonResponse({'error': 'Maximum 10 CVs allowed'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Extract CV texts
            cv_contents = []
            for file in cv_files:
                content = self.extract_text_from_file(file)
                cv_contents.append({
                    'filename': file.name,
                    'content': content
                })

            if all((not item['content'] or not item['content'].strip()) for item in cv_contents):
                return JsonResponse({'error': 'No readable text extracted from uploaded files.'},
                                    status=status.HTTP_400_BAD_REQUEST)

            # Configure Gemini
            if not getattr(settings, 'GEMINI_API_KEY', None):
                return JsonResponse({'error': 'GEMINI_API_KEY not configured on server.'},
                                    status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            genai.configure(api_key=settings.GEMINI_API_KEY)

            # Use deterministic config for reproducible results
            model = genai.GenerativeModel('gemini-1.5-flash')
            generation_config = {
                "temperature": 0,
                "top_p": 1,
                "top_k": 1
            }

            # Build prompt
            prompt = self.craft_prompt(tor, cv_contents)

            # Call Gemini API
            response = model.generate_content(prompt, generation_config=generation_config)
            ai_output = response.text.strip()
            logger.debug("Gemini API output: %s", ai_output)

            # Parse JSON safely
            cleaned_output = ai_output.strip()
            if cleaned_output.startswith('```json') and cleaned_output.endswith('```'):
                cleaned_output = cleaned_output[7:-3].strip()

            if not cleaned_output.startswith('{'):
                first_brace = cleaned_output.find('{')
                last_brace = cleaned_output.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    cleaned_output = cleaned_output[first_brace:last_brace + 1]
                else:
                    logger.error("Invalid JSON format in AI output: %s", cleaned_output)
                    raise ValueError("Invalid JSON format returned by AI.")

            result = json.loads(cleaned_output)
            return JsonResponse(normalize_result(result))

        except json.JSONDecodeError as e:
            logger.error("JSON decode error: %s, Raw output: %s", str(e), ai_output)
            return JsonResponse({'error': 'Invalid JSON response from AI.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            logger.error("Unexpected error: %s", str(e), exc_info=True)
            return JsonResponse({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # -------------------- Helper Functions --------------------

    def extract_text_from_file(self, file: UploadedFile) -> str:
        """Extract raw text from PDF, DOCX, or TXT file"""
        filename = file.name.lower()
        try:
            if filename.endswith('.pdf'):
                reader = PdfReader(file)
                return ''.join([page.extract_text() or '' for page in reader.pages])
            elif filename.endswith('.docx') or filename.endswith('.doc'):
                doc = Document(file)
                return '\n'.join([para.text for para in doc.paragraphs])
            elif filename.endswith('.txt'):
                return file.read().decode('utf-8', errors='ignore')
            else:
                raise ValueError(f'Unsupported file type: {filename}')
        except Exception as e:
            logger.error("Error extracting text from file %s: %s", filename, str(e))
            raise ValueError(f"Failed to extract text from {filename}")

    def craft_prompt(self, tor: str, cv_contents: list) -> str:
        """Crafts the structured deterministic prompt for Gemini API"""

        def esc(text: str) -> str:
            return text.replace('\\', '\\\\').replace('"', '\\"')

        cvs_text = "[" + ", ".join([
            '{"id":"cv' + str(i + 1) + '","file_name":"' + esc(cv['filename']) + '","cv_text":"' + esc(cv['content']) + '"}'
            for i, cv in enumerate(cv_contents)
        ]) + "]"

        template = """
You are a world-class recruitment AI and expert HR analyst. 
Analyze the provided CVs strictly against the Terms of Reference (ToR). 
You must always return **valid JSON only**. No extra text, no markdown.

====================================================
SCORING CRITERIA (Fixed Weights):
General Qualifications - 20%
  - Education - 10%
  - Years of Experience - 10%
Adequacy for the Assignment - 50%
  - Relevant Project Experience - 25%
  - Donor Experience (WB, ADB, etc.) - 15%
  - Regional Experience - 10%
Specific Skills & Competencies - 30%
  - Technical Skills - 15%
  - Language Proficiency - 10%
  - Certifications - 5%
Total Score - 100%
====================================================

RULES:
1. Deterministic: same input → same output.
2. Do not invent information. If missing, score = 0.
3. All scores must be numeric and total 100%.
4. Recommendations: choose **one best candidate** OR state "No candidates are suitable".
5. Each justification must cite explicit CV evidence (≤25 words). If no evidence, justification = "No evidence in CV."
6. Output must be checked 3 times for correctness before finalizing.
7. Ensure all fields in the output schema are present, even if empty.

IMPORTANT RULES FOR SCORING:
- Each criterion must be scored strictly within its maximum weight. 
- Example: Education is out of 10, Experience is out of 10, Project Experience is out of 25, etc.
- The sum of all criteria MUST be exactly out of 100. 
- Do NOT exceed the maximum weight for any criterion.
- Final "total_score" must always be between 0 and 100 (rounded to 2 decimals).

====================================================
OUTPUT FORMAT:
{
  "tor_text": "Full ToR Text",
  "criteria": [
    {"criterion": "Education", "weight": 10},
    {"criterion": "Years of Experience", "weight": 10},
    {"criterion": "Relevant Project Experience", "weight": 25},
    {"criterion": "Donor Experience (WB, ADB, etc.)", "weight": 15},
    {"criterion": "Regional Experience", "weight": 10},
    {"criterion": "Technical Skills", "weight": 15},
    {"criterion": "Language Proficiency", "weight": 10},
    {"criterion": "Certifications", "weight": 5}
  ],
  "candidates": [
    {
      "candidate_name": "string",
      "recommendation": "Suitable | Not Suitable",
      "scores": {
        "general_qualifications": {
          "education": 0.0,
          "years_of_experience": 0.0,
          "total": 0.0
        },
        "adequacy_for_assignment": {
          "relevant_project_experience": 0.0,
          "donor_experience": 0.0,
          "regional_experience": 0.0,
          "total": 0.0
        },
        "specific_skills_competencies": {
          "technical_skills": 0.0,
          "language_proficiency": 0.0,
          "certifications": 0.0,
          "total": 0.0
        },
        "total_score": 0.0
      },
      "summary_justification": {
        "key_strengths": "string",
        "key_weaknesses": "string"
      },
      "detailed_evaluation": [
        {
          "criterion": "string",
          "weight": 0,
          "score": 0.0,
          "justification": "string"
        }
      ]
    }
  ],
  "comparison_matrix": [
    {
      "candidate_name": "string",
      "total_score": 0.0,
      "rank": 1
    }
  ],
  "final_recommendation": {
    "best_candidate": "Name or None",
    "final_decision": "Highly Suitable | Suitable | Not Suitable | None Suitable (if no one is suitable directly say Not Suitable for all cadidates)",
    "justification": {
    "detailed_explanation": "Detailed explanation why best candidate is chosen OR why no one is suitable AND also explain other candidates are not chosen. Must include strengths, weaknesses, and comparison.",
    "why_he": "Why your recommended candidate is the best candidate. Full details explaining strengths, achievements, and suitability.",
    "why_not_others": [
      {
        "candidate_name": "Candidate 1",
        "reason": "Explain why Candidate 1 was not recommended, including weaknesses or lack of match with ToR."
      },
      {
        "candidate_name": "Candidate 2",
        "reason": "Explain why Candidate 2 was not recommended..."
      }
      // Add more candidates here
    ]
  }
}
====================================================

tor_text: <<TOR_TEXT>>
cvs: <<CVS_TEXT>>
"""
        return template.replace('<<TOR_TEXT>>', esc(tor)).replace('<<CVS_TEXT>>', cvs_text)