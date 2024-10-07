import subprocess
import os
import model_compressor  # test 라이브러리
import furiosa_llm_models  # test 라이브러리

def get_git_commit_hash(library):
    # 라이브러리 경로를 가져옴
    lib_path = os.path.dirname(library.__file__)

    try:
        # 라이브러리의 경로에서 Git 커밋 해시를 가져옴
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=lib_path
        ).strip().decode("utf-8")
        return commit_hash
    except Exception as e:
        return f"Error retrieving commit hash: {e}"

# my_library의 Git commit hash 확인
my_libraries = [model_compressor, furiosa_llm_models]

for library in my_libraries:
    commit_hash = get_git_commit_hash(library)
    print(f"Commit hash: {library} {commit_hash}")