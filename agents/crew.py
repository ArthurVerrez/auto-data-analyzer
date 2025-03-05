import time
import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource


@CrewBase
class AnalystCrew:
    def __init__(self, llm_id) -> None:
        self.llm = LLM(model=llm_id)
        self.agents_config = "config/agents.yaml"
        self.tasks_config = "config/tasks.yaml"

    @agent
    def analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config["analyzer"],
            knowledge_sources=[],
            tools=[],
            max_iter=6,
            max_retry_limit=2,
            max_execution_time=60,
            memory=True,
            verbose=True,
        )

    @task
    def analysis_task(self) -> Task:
        return Task(config=self.tasks_config["bar_chart_suggestion"])

    @crew
    def crew(self, step_callback=None, task_callback=None) -> Crew:
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            step_callback=step_callback,
            task_callback=task_callback,
            # output_log_file=f"agents/logs/{time.time()}.log",
        )
