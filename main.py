import os
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dotenv import load_dotenv
import json

# Telegram Bot Library
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    filters,
)

# CrewAI
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# Gemini API
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API keys from environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Available Gemini models
DEFAULT_MODEL = "gemini-pro"

# Custom class for wrapping Gemini with LLM interface compatible with CrewAI
class GeminiLLM:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = genai.GenerativeModel(name=model_name)
        self.supports_functions = True
        self.supports_stop_words = False

    def __call__(self, prompt: str, **kwargs) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            return f"Error generating content: {str(e)}"

    async def agenerate_content(self, prompt: str, **kwargs) -> str:
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.model.generate_content, prompt, kwargs)
            return response.text
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            return f"Error generating content: {str(e)}"

# Custom tools for CrewAI agents
class InternetSearchTool(BaseTool):
    name = "Internet Search"
    description = "Search for information on the internet."

    def _run(self, query: str) -> str:
        llm = GeminiLLM()
        prompt = f"Search the internet for: {query}\nProvide a comprehensive summary of the results."
        return llm(prompt)

class DataAnalysisTool(BaseTool):
    name = "Data Analysis"
    description = "Analyze data and provide insights."

    def _run(self, data: str) -> str:
        llm = GeminiLLM()
        prompt = f"Analyze the following data and provide key insights:\n{data}"
        return llm(prompt)

class ContentCreationTool(BaseTool):
    name = "Content Creation"
    description = "Create high-quality content based on a topic."

    def _run(self, topic: str) -> str:
        llm = GeminiLLM()
        prompt = f"Create high-quality content about: {topic}"
        return llm(prompt)

# Define the CrewAI agents
def create_researcher_agent() -> Agent:
    return Agent(
        role="Research Specialist",
        goal="Find accurate and detailed information on any given topic",
        backstory="Expert researcher with experience in finding reliable information quickly.",
        llm=GeminiLLM(),
        tools=[InternetSearchTool()],
        verbose=True,
        allow_delegation=True,
    )

def create_analyst_agent() -> Agent:
    return Agent(
        role="Data Analyst",
        goal="Analyze data and extract meaningful insights",
        backstory="Expert in interpreting complex information and finding patterns.",
        llm=GeminiLLM(),
        tools=[DataAnalysisTool()],
        verbose=True,
        allow_delegation=True,
    )

def create_writer_agent() -> Agent:
    return Agent(
        role="Content Writer",
        goal="Create engaging and informative content",
        backstory="Skilled writer capable of creating compelling content on any topic.",
        llm=GeminiLLM(),
        tools=[ContentCreationTool()],
        verbose=True,
        allow_delegation=True,
    )

# Task types and their corresponding agent creators
TASK_TYPES = {
    "research": create_researcher_agent,
    "analysis": create_analyst_agent,
    "writing": create_writer_agent,
}

# Store active tasks and crews for each user
user_tasks: Dict[int, Dict[str, Any]] = {}

# Helper functions
async def create_and_run_crew(user_id: int, task_type: str, task_description: str) -> str:
    try:
        primary_agent = TASK_TYPES[task_type]()
        agents = [primary_agent]

        if task_type == "research":
            agents.append(create_writer_agent())
        elif task_type == "writing":
            agents.append(create_researcher_agent())

        task = Task(
            description=task_description,
            agent=primary_agent,
            expected_output="A comprehensive response addressing all aspects of the task."
        )

        crew = Crew(
            agents=agents,
            tasks=[task],
            verbose=True,
            process=Process.sequential,
        )

        user_tasks[user_id] = {
            "crew": crew,
            "task_description": task_description,
            "status": "running"
        }

        result = crew.kickoff()
        user_tasks[user_id]["status"] = "completed"
        user_tasks[user_id]["result"] = result

        return result
    except Exception as e:
        logger.error(f"Error running crew: {e}")
        return f"Error running crew: {str(e)}"

# Telegram bot handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Welcome to the Research Bot! Use /research <topic> to start.")

async def research_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not context.args:
        await update.message.reply_text("Please specify a research topic.")
        return

    topic = " ".join(context.args)
    await update.message.reply_text(f"🔍 Starting research on: '{topic}'...")

    result = await asyncio.to_thread(create_and_run_crew, user_id, "research", f"Research topic: {topic}")

    MAX_MESSAGE_LENGTH = 4096
    if len(result) <= MAX_MESSAGE_LENGTH:
        await update.message.reply_text(f"Research results:\n\n{result}")
    else:
        chunks = [result[i:i + MAX_MESSAGE_LENGTH] for i in range(0, len(result), MAX_MESSAGE_LENGTH)]
        for i, chunk in enumerate(chunks):
            await update.message.reply_text(f"Results (Part {i+1}):\n\n{chunk}")

# Main function to start the bot
def main() -> None:
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("research", research_command))
    application.run_polling()

if __name__ == "__main__":
    main()
