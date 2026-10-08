from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST
from .engine import analyze_code

def home(request):
    return render(request, "analyzer/home.html")

@require_POST
def analyze(request):
    code = request.POST.get("code", "")
    if not code.strip():
        return JsonResponse({"error": "Paste a Python function to analyze."}, status=400)
    if len(code) > 50000:
        return JsonResponse({"error": "Code is limited to 50,000 characters."}, status=413)
    try:
        result = analyze_code(code)
    except Exception as exc:
        return JsonResponse({"error": f"Analysis failed: {exc}"}, status=500)
    return JsonResponse(result)
