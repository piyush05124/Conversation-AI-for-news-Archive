import re
import os
import shutil
import csv
from datetime import datetime
from dateutil.parser import parse
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
# Configuration

CSV_PATH = r"D:\nlp_project_haystack\newsQA_v2.0\IndianFinancialNews.csv"
rebuild_db_status = False  # Set to True to rebuild vector database
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "t5-large"
DB_PATH = "financial_news_db"
COLLECTION_NAME = "indian_financial_news_v1"
MAX_TOKENS = 1024
BATCH_SIZE = 100





class FinancialNewsChatbot:
    def __init__(self, rebuild_db=False):
        self.vector_db = self.initialize_components(rebuild_db)
        self.llm = self.create_llm_pipeline()
        self.qa_chain = self.create_chat_chain()

    def initialize_components(self, rebuild_db):
        """Initialize database with proper error handling"""
        if rebuild_db and os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)

        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        
        try:
            vector_db = Chroma(
                persist_directory=DB_PATH,
                embedding_function=embeddings,
                collection_metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Chroma: {str(e)}")

        if rebuild_db:
            self.ingest_data(vector_db)
            
        return vector_db

    def ingest_data(self, vector_db):
        """Robust CSV ingestion with date validation"""
        docs = []
        row_count = 0
        
        try:
            with open(CSV_PATH, 'r', encoding='utf-8') as csv_file:
                reader = csv.reader(csv_file)
                header = next(reader)  # Skip header
                
                for i, row in enumerate(reader):
                    row_count += 1
                    try:
                        if len(row) < 3:
                            print(f"Row {i+2}: Insufficient columns")
                            continue

                        # Clean and validate data
                        raw_date = self.clean_field(row[0])
                        title = self.clean_field(row[1])
                        desc = self.clean_field(row[2])
                        
                        if not (title and desc):
                            print(f"Row {i+2}: Missing title/description")
                            continue
                            
                        
                        year = self.parse_year(raw_date)
                        
                        docs.append(Document(
                            page_content=f"Title: {title}\nDate: {raw_date}\nDetails: {desc}",
                            metadata={
                                "source": CSV_PATH,
                                "row": i+2,
                                "title": title[:100],
                                "raw_date": raw_date,
                                "year": year
                            }
                        ))

                    except Exception as e:
                        print(f"Row {i+2} error: {str(e)}")
                        continue

                # Batch insert with progress tracking
                total_docs = len(docs)
                for i in range(0, total_docs, BATCH_SIZE):
                    batch = docs[i:i+BATCH_SIZE]
                    vector_db.add_documents(batch)
                    print(f"Inserted {min(i+BATCH_SIZE, total_docs)}/{total_docs} documents")

        except FileNotFoundError:
            raise RuntimeError(f"CSV file not found: {CSV_PATH}")
        except Exception as e:
            raise RuntimeError(f"CSV parsing failed: {str(e)}")

        print(f"\nSuccessfully ingested {len(docs)}/{row_count} rows")
        return vector_db

    def clean_field(self, value):
        """Safe string cleaning"""
        return str(value).strip('"').strip() if value else ""

    def parse_year(self, date_str):
        """Robust year parsing with multiple fallbacks"""
        try:
            # Try ISO format first
            dt = datetime.fromisoformat(date_str)
            return dt.year
        except ValueError:
            try:
                # Try dateutil parse
                dt = parse(date_str, fuzzy=True)
                return dt.year
            except:
                # Fallback to manual year extraction
                match = re.search(r'\b\d{4}\b', date_str)
                return int(match.group(0)) if match else None

    def create_llm_pipeline(self):
        """Safe LLM initialization"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
            model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
            
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=1024,
                temperature=0.3,
                device_map="auto"
            )
            return HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            raise RuntimeError(f"LLM initialization failed: {str(e)}")

    def create_chat_chain(self):
        """Create updated conversation chain"""
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_db.as_retriever(search_kwargs={"k": 5}),
            memory=memory,
            return_source_documents=True,
            get_chat_history=lambda h: h,
            verbose=True
        )

    def query(self, question):
        """Safe query handling"""
        try:
            result = self.qa_chain({"question": question})
            return self.format_response(result)
        except Exception as e:
            return f"Error: {str(e)}"

    def format_response(self, result):
        """Error-resistant formatting"""
        try:
            response = result.get("answer", "No answer found")
            sources = []
            
            for doc in result.get("source_documents", [])[:3]:
                meta = doc.metadata
                source_info = f"{meta.get('title', 'Untitled')} ({meta.get('raw_date', 'Unknown date')})"
                sources.append(f"- {source_info}")
            
            return f"{response}\n\nSources:\n" + "\n".join(sources)
        except Exception as e:
            return f"Response formatting error: {str(e)}"

# ====================================================Usage example
# if __name__ == "__main__":
#     try:
#         chatbot = FinancialNewsChatbot(rebuild_db=rebuild_db_status)
#         print(chatbot.query("Explain reverse repo rate mechanics"))
#     except Exception as e:
#         print(f"Startup failed: {str(e)}")

#========================================================================
import streamlit as st

# ... (keep all your original imports here) ...

# Configuration and FinancialNewsChatbot class remain unchanged
# [Keep all the original code here until the main block]

def main():
    st.set_page_config(page_title="Financial News Chatbot", page_icon="💬")
    st.title("Financial News Chatbot 💬📈")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Configuration")
        rebuild_db = st.checkbox("Rebuild Database", help="WARNING: This will delete and recreate the vector database")
        st.markdown("---")
        st.markdown("## About")
        st.markdown("This chatbot answers questions about Indian financial news using AI-powered semantic search.")
    
    # Initialize chatbot with caching
    @st.cache_resource(show_spinner=False)
    def load_chatbot(rebuild_flag):
        try:
            if rebuild_flag:
                with st.status("Rebuilding database...", expanded=True) as status:
                    st.write("Processing CSV data...")
                    chatbot = FinancialNewsChatbot(rebuild_db=True)
                    status.update(label="Database rebuilt!", state="complete")
            else:
                with st.spinner("Loading chatbot..."):
                    chatbot = FinancialNewsChatbot(rebuild_db=False)
            return chatbot
        except Exception as e:
            st.error(f"Initialization failed: {str(e)}")
            return None

    chatbot = load_chatbot(rebuild_db)
    
    if chatbot is None:
        st.stop()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about financial news"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing financial data..."):
                try:
                    response = chatbot.query(prompt)
                    # Split response into answer and sources
                    if "Sources:\n" in response:
                        answer, sources = response.split("Sources:\n", 1)
                        st.markdown(answer)
                        with st.expander("View sources"):
                            st.markdown(sources)
                    else:
                        st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()