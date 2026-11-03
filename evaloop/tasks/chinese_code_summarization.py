"""Chinese code-summarization task used by the cross-lingual loop."""

from .code_summarization import CodeSummarizationTask
from .factory import register_task


@register_task("chinese_code_summarization")
class ChineseCodeSummarizationTask(CodeSummarizationTask):
    """Describe generated Python code in Chinese for the next loop cycle.

    The inherited ``postprocess`` method deliberately re-attaches the assert
    line from the cycle's input.  This keeps the function name and call
    signature stable while changing only the language of the model-generated
    description.
    """

    def create_prompt(self, input_text: str) -> str:
        """Build a prompt that requires a Chinese-only task description."""
        prompt_template = self.task_config.get(
            "prompt_template",
            "用一句中文总结以下 Python 代码实现的任务，并以“编写一个 Python 函数来”开头：\n\n"
            "```python\n{code}\n```\n\n"
            "只返回这句中文总结，不要包含代码、Markdown 或额外解释。",
        )
        return prompt_template.format(code=input_text)
