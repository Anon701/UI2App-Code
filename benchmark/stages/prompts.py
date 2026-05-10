"""
UI2App prompt templates — Plan / Generate / Self-Debug stages.

These are the prompts referenced verbatim in the paper appendix (§A.prompts).
They live here as builder functions rather than constants so f-string
interpolation (slicing, line-joining) stays inside the prompt definition.

Each function returns a fully-rendered prompt string.
"""


def make_plan_prompt(M):
    """Stage 1 (Plan): asks the model to emit a JSON file plan + extra_dependencies.

    Args:
        M: number of input screenshots.
    """
    return f"""You are an expert frontend developer. You are given {M} screenshots of a web application. Your task is to reproduce the UI as a complete, runnable Vite + React + TypeScript + Tailwind CSS project.

The project scaffold has already been set up (you saw it above). src/App.tsx is the root component mounted by main. All source files you generate must be under src/.

tsconfig uses "jsx": "react-jsx" so explicit React imports are unnecessary. react-router-dom and lucide-react are available in dependencies.

<code_guidelines>
- Use coding best practices.
- Use Tailwind CSS utility classes for styling. Match the screenshots as closely as possible.
- Use relative imports. Do not add file extensions in imports for .ts/.tsx files.
- Files containing JSX must use .tsx extension.
</code_guidelines>

Analyze the {M} screenshots carefully. Then output a JSON file plan — a list of all source files you will create, with a one-line description of each file's purpose.

Output ONLY valid JSON (no markdown fences, no explanation):

{{"extra_dependencies": {{}}, "plan": [{{"path": "src/App.tsx", "description": "Root component with routing setup"}}, ...]}}

Rules:
- Include src/App.tsx and all component/page files. All paths start with "src/".
- "extra_dependencies" lists any npm packages beyond the scaffold.
- Do NOT include scaffold files. Do NOT generate any code yet — only the plan."""


def make_gen_prompt(plan_summary):
    """Stage 2 (Generate): asks the model to emit complete source for every file in the plan.

    Args:
        plan_summary: pre-formatted multi-line string of "  - path: description" entries.
    """
    return f"""Now generate the COMPLETE source code for ALL files in the plan below.

<project_plan>
{plan_summary}
</project_plan>

Output each file using this exact format (no markdown fences, no explanation):

--- src/App.tsx ---
[complete source code for App.tsx]

--- src/components/Navbar.tsx ---
[complete source code for Navbar.tsx]

... and so on for every file in the plan.

CRITICAL RULES:
- Write COMPLETE file content for every file. Never truncate, abbreviate, or use comments like "// rest of code".
- Files containing JSX/TSX syntax (<div>, <Component/>) MUST use .tsx extension.
- Use 2 spaces for indentation.
- Every file must start with "--- filepath ---" on its own line.
- Do NOT output anything other than the file blocks."""


def make_locate_prompt(errors, file_list):
    """Stage 3a (Self-Debug, locate): when regex can't extract the failing file, ask the model.

    Args:
        errors: build error stderr (will be truncated to 2000 chars).
        file_list: list[str] of source file relative paths.
    """
    return f"""The following build error occurred. Which source file(s) need to be fixed?

<error>
{errors[:2000]}
</error>

<project_files>
{chr(10).join(file_list)}
</project_files>

Reply with ONLY the file path(s) that need fixing, one per line. No explanation."""


def make_runtime_fix_prompt(errors, file_contents):
    """Stage 3b (Self-Debug, runtime fix): build passes but home route renders blank.

    Args:
        errors: runtime error text (will be truncated to 2000 chars).
        file_contents: list[str] of pre-formatted <file path="..."> blocks.
    """
    return f"""The React app builds successfully but renders a blank page at runtime.

<error>
{errors[:2000]}
</error>

{chr(10).join(file_contents)}

For EACH file that needs changes, output:
--- filepath ---
<complete fixed file content>

Output ONLY the fixed files — no explanation, no markdown fences."""


def make_build_fix_prompt(fpath, errors, file_listing, current_code):
    """Stage 3c (Self-Debug, build fix): single-file fix for a build error.

    Args:
        fpath: relative path of the file to fix.
        errors: build error stderr (will be truncated to 2000 chars).
        file_listing: newline-separated source file listing.
        current_code: current contents of the file being fixed.
    """
    return f"""Fix the following build error in {fpath}:

<error>
{errors[:2000]}
</error>

<project_files>
{file_listing}
</project_files>

<current_code file="{fpath}">
{current_code}
</current_code>

Output ONLY the fixed complete file content — no explanation, no markdown fences."""
