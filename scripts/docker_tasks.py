import subprocess
import os

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

def llm_up():
    run_command(["docker", "compose", "-f", "docker-compose.llm.yml", "up", "-d", "--remove-orphans"])

def llm_down():
    run_command(["docker", "compose", "-f", "docker-compose.llm.yml", "down"])

def trace_up():
    run_command(["docker", "compose", "-f", "docker-compose.trace.yml", "up", "-d", "--remove-orphans"])

def trace_down():
    run_command(["docker", "compose", "-f", "docker-compose.trace.yml", "down"])

def app_up():
    run_command(["docker", "compose", "up", "-d", "--remove-orphans"])

def app_build():
    run_command(["docker", "compose", "up", "--build", "-d", "--remove-orphans"])

def app_down():
    run_command(["docker", "compose", "down"])

def start_all():
    print("Starting all services...")
    llm_up()
    trace_up()
    app_up()

def stop_all():
    print("Stopping all services...")
    app_down()
    trace_down()
    llm_down()

def status():
    run_command(["docker", "compose", "ps"])
    run_command(["docker", "compose", "-f", "docker-compose.llm.yml", "ps"])
    run_command(["docker", "compose", "-f", "docker-compose.trace.yml", "ps"])
