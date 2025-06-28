import os
import json
import subprocess
import re
import shutil
from pathlib import Path
import tempfile

def group_issues_by_repository(dataset):
    """Group all issues by repository and SHA for efficient processing."""
    repo_issues = {}

    for entry in dataset:
        repo_url = entry['repo_url']
        sha = entry['before_fix_sha']

        if repo_url not in repo_issues:
            repo_issues[repo_url] = {}

        if sha not in repo_issues[repo_url]:
            repo_issues[repo_url][sha] = []

        repo_issues[repo_url][sha].append(entry)

    return repo_issues

def clone_repository(repo_url, temp_dir):
    """Clone a repository to a temporary directory."""
    repo_name = repo_url.split('/')[-1]
    permanent_repo_path = Path("../data/repos") / repo_name

    if permanent_repo_path.exists():
        print(f"Repository {repo_name} found in ../data/repos/, using local copy")
        try:
            subprocess.run(["git", "clone", str(permanent_repo_path), temp_dir],
                           check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error cloning from local repository: {e}. Trying remote...")

    try:
        print(f"Cloning {repo_url}...")
        subprocess.run(["git", "clone", repo_url, temp_dir],
                       check=True, capture_output=True)

        if not permanent_repo_path.exists():
            permanent_repo_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Cloning to permanent location: {permanent_repo_path}")
            subprocess.run(["git", "clone", repo_url, permanent_repo_path],
                           check=True, capture_output=True)

        return True
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        return False

def checkout_commit(repo_dir, sha):
    """Checkout a specific commit in a repository."""
    try:
        print(f"  Checking out SHA: {sha[:7]}...")
        subprocess.run(["git", "-C", repo_dir, "checkout", sha, "-q"],
                       check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error checking out commit: {e}")
        return False

def extract_java_functions(content):
    """Extract function information from Java content."""
    functions = []
    method_pattern = r'(public|private|protected|static|\s) +[\w\<\>\[\]]+\s+(\w+) *\([^\)]*\) *\{?'
    matches = re.finditer(method_pattern, content)

    for match in matches:
        method_start_line = content[:match.start()].count('\n') + 1
        method_name = match.group(2)
        open_braces = 0
        close_pos = match.end()

        for i in range(match.end(), len(content)):
            if content[i] == '{':
                open_braces += 1
            elif content[i] == '}':
                if open_braces == 0:
                    close_pos = i
                    break
                open_braces -= 1

        method_end_line = content[:close_pos].count('\n') + 1

        functions.append({
            'name': method_name,
            'start_line': method_start_line,
            'end_line': method_end_line
        })

    return functions

def extract_functions(content, language):
    """Extract function information based on language."""
    if language == "java":
        return extract_java_functions(content)
    return []

def analyze_file_structure(file_path):
    """Analyze a file to identify functions and global code regions."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.splitlines()
            total_lines = len(lines)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return {"exists": False}

    file_ext = os.path.splitext(file_path)[1].lower()
    language = "java"
    if file_ext == ".py":
        language = "python"
    elif file_ext in [".js", ".ts"]:
        language = "javascript"
    elif file_ext in [".c", ".cpp", ".h", ".hpp"]:
        language = "c"

    functions = extract_functions(content, language)

    all_lines = set(range(1, total_lines + 1))
    for func in functions:
        func_lines = set(range(func['start_line'], func['end_line'] + 1))
        all_lines -= func_lines

    global_lines = sorted(list(all_lines))

    global_spans = []
    if global_lines:
        span_start = global_lines[0]
        prev_line = global_lines[0]

        for line in global_lines[1:]:
            if line > prev_line + 1:
                global_spans.append([span_start, prev_line])
                span_start = line
            prev_line = line

        global_spans.append([span_start, prev_line])

    return {
        "exists": True,
        "functions": functions,
        "global_lines": global_lines,
        "global_spans": global_spans,
        "file_content": lines
    }

def update_issue_with_file_info(issue, file_info, line_num):
    """Update an issue with information about the file and containing function."""
    if not file_info.get("exists", False):
        issue['buggy_function_name'] = "file_not_found"
        issue['buggy_function_span'] = [0, 0]
        return

    if line_num > len(file_info["file_content"]):
        issue['buggy_function_name'] = "line_number_invalid"
        issue['buggy_function_span'] = [0, 0]
        return

    for func in file_info["functions"]:
        if func['start_line'] <= line_num <= func['end_line']:
            issue['buggy_function_name'] = func['name']
            issue['buggy_function_span'] = [func['start_line'], func['end_line']]
            return

    for span in file_info["global_spans"]:
        if span[0] <= line_num <= span[1]:
            issue['buggy_function_name'] = "global"
            issue['buggy_function_span'] = span
            return

    issue['buggy_function_name'] = "global"
    issue['buggy_function_span'] = [line_num, line_num]

def process_repositories(repo_issues):
    """Process each repository once, handling all its issues."""
    temp_root = Path("./temp_repos")
    temp_root.mkdir(exist_ok=True)

    try:
        for repo_url, sha_issues in repo_issues.items():
            repo_name = repo_url.split('/')[-1]
            print(f"\nProcessing repository: {repo_name}")

            temp_dir = temp_root / repo_name
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

            if not clone_repository(repo_url, temp_dir):
                print(f"Skipping repository {repo_name} due to clone error")
                continue

            for sha, issues in sha_issues.items():
                print(f"  Processing SHA: {sha[:7]} with {len(issues)} issues")

                if not checkout_commit(temp_dir, sha):
                    print(f"  Skipping SHA {sha[:7]} due to checkout error")
                    continue

                file_cache = {}

                for issue in issues:
                    if 'buggy_file_name' not in issue or 'path_to_buggy_file' not in issue:
                        continue

                    file_path = os.path.join(
                        issue['path_to_buggy_file'],
                        issue['buggy_file_name']
                    )

                    if file_path not in file_cache:
                        full_path = os.path.join(temp_dir, file_path)
                        if os.path.exists(full_path):
                            try:
                                file_cache[file_path] = analyze_file_structure(full_path)
                            except Exception as e:
                                print(f"  Error analyzing file {file_path}: {e}")
                                file_cache[file_path] = {"exists": False}
                        else:
                            print(f"  File not found: {file_path}")
                            file_cache[file_path] = {"exists": False}

                    update_issue_with_file_info(
                        issue,
                        file_cache[file_path],
                        issue.get('buggy_line_number', 0)
                    )
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root)

    return repo_issues

def main():
    """Main function to process the dataset."""
    print("Starting repository analysis for bug localization")

    input_file = "../data/bug_localization_dataset.json"
    output_file = "../data/bug_localization_dataset_updated.json"

    print(f"Loading dataset from {input_file}...")
    try:
        with open(input_file, 'r') as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("Grouping issues by repository and commit...")
    repo_issues = group_issues_by_repository(dataset)

    repo_count = len(repo_issues)
    issue_count = sum(
        len(issues) for repo in repo_issues.values()
        for issues in repo.values()
    )
    sha_count = sum(len(repo) for repo in repo_issues.values())

    print(f"Found {issue_count} issues across {repo_count} repositories and {sha_count} commits")
    print("Processing repositories...")
    updated_issues = process_repositories(repo_issues)

    updated_dataset = [
        issue for repo in repo_issues.values()
        for sha_issues in repo.values()
        for issue in sha_issues
    ]

    updated_count = sum(
        1 for issue in updated_dataset
        if issue.get('buggy_function_name') in ['global', 'file_not_found', 'line_number_invalid']
        or (issue.get('buggy_function_name') and issue['buggy_function_name'] != '')
    )

    print(f"\nUpdated {updated_count} out of {len(updated_dataset)} entries with function information")
    print(f"Saving updated dataset to {output_file}...")
    try:
        with open(output_file, 'w') as f:
            json.dump(updated_dataset, f, indent=2)
        print("Done.")
    except Exception as e:
        print(f"Error saving dataset: {e}")

if __name__ == "__main__":
    main()
