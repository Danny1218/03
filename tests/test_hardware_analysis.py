# Set working directory to project root and update sys.path
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root)
sys.path.insert(0, project_root)

from src.hardware_analysis import analyze_hardware

def test_hardware_analysis(capsys):
    analyze_hardware()
    out = capsys.readouterr().out
    assert 'OS:' in out 