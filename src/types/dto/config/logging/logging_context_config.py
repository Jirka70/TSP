from pydantic import BaseModel


class LoggingContextConfig(BaseModel):
    run_id: bool = True
    git_commit_hash: bool = True
    pipeline_name: bool = True
    step: bool = True