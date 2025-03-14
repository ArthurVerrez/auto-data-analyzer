from calendar import week
import time
import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from pydantic import BaseModel
from enum import Enum
from typing import List, Optional


class YAggEnum(str, Enum):
    distinct_values = "distinct_values"
    sum = "sum"
    record_count = "record_count"
    median = "median"


class OrderEnum(str, Enum):
    asc = "asc"
    desc = "desc"
    rand = "rand"


class TimeTruncEnum(str, Enum):
    year = "year"
    month = "month"
    week = "week"
    day = "day"
    hour = "hour"
    minute = "min"
    second = "s"


class BarChartSpec(BaseModel):
    title: str
    x: str
    y: str
    y_agg: YAggEnum
    x_order: Optional[OrderEnum] = None
    y_order: Optional[OrderEnum] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None


class LineChartSpec(BaseModel):
    title: str
    x: str
    y: str
    y_agg: YAggEnum
    x_order: Optional[OrderEnum] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    x_time_trunc: Optional[TimeTruncEnum] = None


class HistogramSpec(BaseModel):
    title: str
    x: str
    bins: int
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    y_agg: Optional[YAggEnum] = None


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
    def bar_chart_suggestion_task(self) -> Task:
        return Task(
            config=self.tasks_config["bar_chart_suggestion"], output_json=BarChartSpec
        )

    @task
    def line_chart_suggestion_task(self) -> Task:
        return Task(
            config=self.tasks_config["line_chart_suggestion"], output_json=LineChartSpec
        )

    @task
    def histogram_suggestion_task(self) -> Task:
        return Task(
            config=self.tasks_config["histogram_suggestion"], output_json=HistogramSpec
        )

    @crew
    def crew(self, step_callback=None, task_callback=None) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            step_callback=step_callback,
            task_callback=task_callback,
            # output_log_file=f"agents/logs/{time.time()}.log",
        )
