#!/usr/bin/env python3
"""
Test script to verify the Simpsons RAG chatbot setup.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import scrapy
        print("✓ Scrapy imported successfully")
    except ImportError as e:
        print(f"✗ Scrapy import failed: {e}")
        return False
    
    try:
        import fastapi
        print("✓ FastAPI imported successfully")
    except ImportError as e:
        print(f"✗ FastAPI import failed: {e}")
        return False
    
    try:
        import langchain
        print("✓ LangChain imported successfully")
    except ImportError as e:
        print(f"✗ LangChain import failed: {e}")
        return False
    
    try:
        import chromadb
        print("✓ ChromaDB imported successfully")
    except ImportError as e:
        print(f"✗ ChromaDB import failed: {e}")
        return False
    
    try:
        import ctransformers
        print("✓ ctransformers imported successfully")
    except ImportError as e:
        print(f"✗ ctransformers import failed: {e}")
        return False
    
    try:
        import openai
        print("✓ OpenAI imported successfully")
    except ImportError as e:
        print(f"✗ OpenAI import failed: {e}")
        return False
    
    return True

def test_file_structure():
    """Test if all required files and directories exist"""
    print("\nTesting file structure...")
    
    required_files = [
        "requirements.txt",
        "README.md",
        "rag_chatbot.py",
        "run_scraper.py",
        "simpsons_scraper/simpsons_scraper/spiders/episodes.py",
        "simpsons_scraper/simpsons_scraper/pipelines.py",
        "simpsons_scraper/simpsons_scraper/settings.py",
        "templates/index.html"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def test_data_files():
    """Test if episode data files exist"""
    print("\nTesting data files...")
    
    data_files = [
        "data/simpsons_episodes.csv",
        "data/simpsons_episodes.json"
    ]
    
    data_exists = False
    for file_path in data_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
            data_exists = True
        else:
            print(f"⚠ {file_path} - Not found (run scraper first)")
    
    return data_exists

def test_scrapy_spider():
    """Test if Scrapy spider can be imported"""
    print("\nTesting Scrapy spider...")
    
    try:
        # Add simpsons_scraper to path
        sys.path.insert(0, str(Path(__file__).parent / 'simpsons_scraper'))
        
        from simpsons_scraper.spiders.episodes import SimpsonsEpisodesSpider
        print("✓ SimpsonsEpisodesSpider imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import spider: {e}")
        return False

def main():
    """Main test function"""
    print("Simpsons RAG Chatbot - Setup Test")
    print("=" * 40)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test file structure
    structure_ok = test_file_structure()
    
    # Test data files
    data_ok = test_data_files()
    
    # Test Scrapy spider
    spider_ok = test_scrapy_spider()
    
    print("\n" + "=" * 40)
    print("Test Results:")
    print(f"Imports: {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"File Structure: {'✓ PASS' if structure_ok else '✗ FAIL'}")
    print(f"Data Files: {'✓ PASS' if data_ok else '⚠ WARNING (run scraper)'}")
    print(f"Scrapy Spider: {'✓ PASS' if spider_ok else '✗ FAIL'}")
    
    if imports_ok and structure_ok and spider_ok:
        print("\n🎉 Setup looks good! You can now:")
        if not data_ok:
            print("1. Run: python run_scraper.py")
        print("2. Run: python rag_chatbot.py")
        print("3. Open: http://localhost:8000")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 