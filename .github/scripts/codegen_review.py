import os
import sys
import json
import time

try:
    from codegen.agents.agent import Agent
except ImportError:
    print("Error: The 'codegen' package is not installed. Please add 'pip install codegen' to your workflow.", file=sys.stderr)
    sys.exit(1)

def main():
    # Get environment variables from the GitHub Actions environment
    org_id = os.getenv('CODEGEN_ORG_ID')
    api_token = os.getenv('CODEGEN_API_TOKEN')
    repo_name = os.getenv('GITHUB_REPOSITORY', 'N/A')
    event_path = os.getenv('GITHUB_EVENT_PATH')

    if not all([org_id, api_token]):
        print("Error: Missing required environment variables (CODEGEN_ORG_ID, CODEGEN_API_TOKEN).", file=sys.stderr)
        sys.exit(1)

    if not event_path:
        print("Error: GITHUB_EVENT_PATH not set. This script must be run in a GitHub Actions context.", file=sys.stderr)
        sys.exit(1)

    # Extract PR number from the event payload
    pr_number = 'N/A'
    try:
        with open(event_path, 'r') as f:
            event_data = json.load(f)
            if 'pull_request' in event_data:
                pr_number = event_data['pull_request'].get('number', 'N/A')
            else:
                pr_number = event_data.get('inputs', {}).get('pr_number', 'N/A')
    except (IOError, json.JSONDecodeError) as e:
        print(f"Warning: Could not read PR number from GITHUB_EVENT_PATH: {e}", file=sys.stderr)

    # Initialize the real Codegen agent
    try:
        agent = Agent(org_id=org_id, token=api_token)
    except Exception as e:
        print(f"Error initializing Codegen Agent: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Create the review prompt for the agent
    prompt = f"""
Please perform a comprehensive review of Pull Request #{pr_number} in the repository {repo_name}.

Focus your analysis on the following areas:
- **Code Quality & Best Practices:** Adherence to coding standards, clarity, and maintainability.
- **Potential Bugs & Security Issues:** Identify logical errors, edge cases, and common vulnerabilities (e.g., injection, XSS).
- **Performance Considerations:** Look for inefficient algorithms, database queries, or resource usage patterns.
- **Documentation & Testing:** Assess the quality of comments, READMEs, and the adequacy of test coverage.

Provide specific, actionable feedback. Use line-by-line comments where appropriate to pinpoint exact locations for improvement.
"""
    
    print(f"Submitting review request for PR #{pr_number} in {repo_name}...")

    try:
        # Run the agent to start the review task
        task = agent.run(prompt=prompt)
        
        # Poll for completion status with a timeout
        max_attempts = 30  # Poll for 5 minutes (30 attempts * 10 seconds)
        attempt = 0
        
        while attempt < max_attempts:
            task.refresh()
            print(f"Task status: {task.status} (Attempt {attempt + 1}/{max_attempts})")
            
            if task.status == "completed":
                print("✅ Review completed successfully!")
                if hasattr(task, 'result') and task.result:
                    print(f"Result: {task.result}")
                break
            elif task.status == "failed":
                print(f"❌ Review task failed. Error: {getattr(task, 'error', 'Unknown error')}", file=sys.stderr)
                sys.exit(1)
            
            attempt += 1
            time.sleep(10)
        
        if attempt >= max_attempts:
            print("⏰ Review timed out after 5 minutes.", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"An unexpected error occurred while running the Codegen review: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
