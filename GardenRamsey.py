import os
import streamlit as st
from typing import List, Dict, Any
import json
import tempfile
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import retrieval_qa
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType

# Environment setup
# Use environment variable or Streamlit secrets instead of hardcoding
# Replace with your Gemini API key through environment variables
API_KEY = os.environ.get("GOOGLE_API_KEY", "")

class RecipeRecommendationChatbot:
    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # Initialize Gemini model
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.7)
        self.search_tool = DuckDuckGoSearchRun()
        self.recipe_vectorstore = None
        # Use HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
    def load_recipe_books(self, uploaded_files):
        """Load and index user-provided recipe books from Streamlit uploaded files."""
        documents = []
        
        for uploaded_file in uploaded_files:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                if tmp_file_path.endswith('.pdf'):
                    loader = PyPDFLoader(tmp_file_path)
                else:
                    loader = TextLoader(tmp_file_path)
                documents.extend(loader.load())
            finally:
                # Clean up the temp file
                os.unlink(tmp_file_path)
        
        if not documents:
            return False
            
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(documents)
        
        # Using FAISS vector store
        self.recipe_vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=self.embeddings
        )
        
        return len(splits)
    
    def get_recipe_recommendation(self, ingredients: List[str], cuisine_type: str = None, 
                                  dietary_restrictions: List[str] = None) -> str:
        """Generate recipe recommendations based on provided ingredients."""
        # Format ingredients for the prompt
        ingredients_str = ", ".join(ingredients)
        
        # Build prompt depending on what knowledge sources we have
        if self.recipe_vectorstore:
            # Create a RAG-based recommendation using the vector database
            retriever = self.recipe_vectorstore.as_retriever(
                search_kwargs={"k": 3}  # Reduced from 5 to 3 for better precision
            )
            
            # Add optional parameters if provided
            cuisine_type_info = f"Cuisine Type: {cuisine_type}" if cuisine_type else ""
            dietary_info = f"Dietary Restrictions: {', '.join(dietary_restrictions)}" if dietary_restrictions else ""
            
            prompt_template = """
            You are a helpful cooking assistant that recommends recipes based on available ingredients.
            
            Available Ingredients: {ingredients}
            {cuisine_type_info}
            {dietary_info}
            
            Based on the above information and the following relevant recipes from the user's recipe collection, 
            suggest a detailed recipe the user can make.
            
            {context}
            
            The recipe should include:
            1. A creative name for the dish
            2. List of ingredients with amounts
            3. Step-by-step cooking instructions
            4. Approximate cooking time
            5. Serving size
            
            If the ingredients are insufficient, suggest minimal additions needed.
            """
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["ingredients", "context", "cuisine_type_info", "dietary_info"]
            )
            
            # Use try-except to handle potential issues with RAG
            try:
                qa_chain = retrieval_qa.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": PROMPT}
                )
                
                # Fixed: Use invoke instead of get_relevant_documents
                query = f"What recipe can I make with these ingredients: {ingredients_str}?"
                # Use the retriever's invoke method
                docs = retriever.invoke(query)
                context_text = "/n/n".join([doc.page_content for doc in docs])
                
                # Make sure to pass all required parameters to the chain
                result = retrieval_qa.invoke({
                    "query": query,
                    "ingredients": ingredients_str,
                    "cuisine_type_info": cuisine_type_info,
                    "dietary_info": dietary_info,
                    "context": context_text
                })
                
                return result["result"]
            except Exception as e:
                # Fallback to direct Gemini if something goes wrong with RAG
                st.error(f"Error using recipe database: {str(e)}. Falling back to general recommendations.")
                return self._direct_gemini_recipe(ingredients_str, cuisine_type, dietary_restrictions)
        else:
            # Use direct Gemini API when no user recipes are available
            return self._direct_gemini_recipe(ingredients_str, cuisine_type, dietary_restrictions)
    
    def _direct_gemini_recipe(self, ingredients_str, cuisine_type=None, dietary_restrictions=None):
        """Generate recipe using direct Gemini API call (no RAG)."""
        try:
            genai.configure(api_key=API_KEY)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            prompt = f"""
            You are a helpful cooking assistant that recommends recipes based on available ingredients.
            
            Available Ingredients: {ingredients_str}
            """
            
            if cuisine_type:
                prompt += f"/nCuisine Type: {cuisine_type}"  
                
            if dietary_restrictions:
                prompt += f"/nDietary Restrictions: {', '.join(dietary_restrictions)}"  
                
            prompt += """
            
            Suggest a detailed recipe the user can make with these ingredients.
            
            The recipe should include:
            1. A creative name for the dish
            2. List of ingredients with amounts
            3. Step-by-step cooking instructions
            4. Approximate cooking time
            5. Serving size
            
            If the ingredients are insufficient, suggest minimal additions needed.
            """
            
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            # Handle the API key error more gracefully
            error_message = str(e)
            if "API key" in error_message:
                return "I'm having trouble connecting to the recipe service. Please make sure you've configured a valid API key."
            else:
                return f"I'm having trouble connecting to the recipe service. Please try again. Error: {error_message}"
    
    def chat(self, user_input: str) -> str:
        """Process general user chat inputs that aren't recipe requests."""
        # Use direct Gemini API for chatting
        try:
            genai.configure(api_key=API_KEY)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            prompt = f"""
            You are a helpful cooking assistant. 
            
            User message: {user_input}
            
            Provide a helpful response related to cooking, recipes, or food preparation.
            """
            
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            # Handle the API key error more gracefully
            error_message = str(e)
            if "API key" in error_message:
                return "I'm having trouble connecting to the chat service. Please make sure you've configured a valid API key."
            else:
                return f"I'm having trouble connecting to the chat service. Please try again. Error: {error_message}"

# Streamlit UI
def main():
    # Fix for asyncio and torch errors - set environment variable to disable eager execution
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["STREAMLIT_SERVER_WATCH_EXCLUDE_PATTERNS"] = "*torch*"
    
    st.set_page_config(page_title="Garden Ramsey Welcomes you.", page_icon="🍳")
    
    # Initialize session state
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = RecipeRecommendationChatbot()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "recipe_books_loaded" not in st.session_state:
        st.session_state.recipe_books_loaded = False
    
    # App header
    st.title("🍳 Tell Garden Ramsey")
    st.subheader("Tell me what ingredients you have, and I'll suggest a recipe!")
    
    # Warn about API key configuration
    if not API_KEY:
        st.warning("⚠️ Please set your Gemini API key as an environment variable 'GOOGLE_API_KEY' before using the application.")
    
    # Sidebar for recipe book uploads
    with st.sidebar:
        st.header("Recipe Book Upload (Optional)")
        st.write("Upload your personal recipe books to get more personalized recommendations.")
        
        uploaded_files = st.file_uploader("Upload recipe books (PDF/TXT)", 
                                        accept_multiple_files=True, 
                                        type=["pdf", "txt"])
        
        if uploaded_files and st.button("Process Recipe Books"):
            with st.spinner("Processing your recipe books..."):
                try:
                    num_chunks = st.session_state.chatbot.load_recipe_books(uploaded_files)
                    if num_chunks:
                        st.session_state.recipe_books_loaded = True
                        st.success(f"Successfully loaded {num_chunks} recipe chunks from your books!")
                    else:
                        st.error("Could not process the uploaded files. Please check the file format.")
                except Exception as e:
                    st.error(f"Error processing recipe books: {str(e)}")
    
    # Status indicator for recipe books
    if st.session_state.recipe_books_loaded:
        st.sidebar.success("✅ Recipe books loaded successfully")
    
    # Ingredient input section
    st.header("What's in Your Kitchen?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ingredients = st.text_area("Enter your ingredients (comma-separated):", 
                                height=100, 
                                placeholder="Example: chicken, rice, onions, garlic")
    
    with col2:
        cuisine_type = st.text_input("Cuisine type (optional):", 
                                    placeholder="Example: Italian, Asian, Mexican")
        
        dietary_restrictions = st.text_input("Dietary restrictions (optional, comma-separated):", 
                                           placeholder="Example: vegetarian, gluten-free, dairy-free")
    
    # Get recipe button
    if st.button("Find Recipe"):
        if not ingredients:
            st.warning("Please enter at least a few ingredients.")
        else:
            with st.spinner("Cooking up a recipe for you..."):
                try:
                    # Process inputs
                    ingredients_list = [i.strip() for i in ingredients.split(",") if i.strip()]
                    dietary_list = [d.strip() for d in dietary_restrictions.split(",") if d.strip()] if dietary_restrictions else None
                    
                    # Get recipe recommendation
                    recipe = st.session_state.chatbot.get_recipe_recommendation(
                        ingredients_list, 
                        cuisine_type, 
                        dietary_list
                    )
                    
                    # Add to chat history
                    st.session_state.messages.append({"role": "user", "content": f"I have these ingredients: {ingredients}"})
                    st.session_state.messages.append({"role": "assistant", "content": recipe})
                except Exception as e:
                    st.error(f"Something went wrong: {str(e)}")
    
    # Chat input
    st.header("Chat with the Recipe Assistant")
    chat_input = st.text_input("Ask anything about cooking:", 
                             placeholder="How do I properly sear a steak?")
    
    if chat_input:
        try:
            response = st.session_state.chatbot.chat(chat_input)
            st.session_state.messages.append({"role": "user", "content": chat_input})
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error processing your request: {str(e)}")
    
    # Display latest chat message in a separate box
    if len(st.session_state.messages) >= 2:
        st.header("Latest Exchange")
        with st.container():
            # Create a styled container for the latest exchange
            st.markdown("---")
            latest_user_msg = st.session_state.messages[-2]['content']
            latest_bot_msg = st.session_state.messages[-1]['content']
            
            # Using expander for user message (but auto-expanded)
            with st.expander("You asked:", expanded=True):
                st.markdown(f"{latest_user_msg}")
            
            # Using a colored container for assistant response
            st.markdown("""
            <style>
            .latest-response {
                background-color: #585554;
                border-left: 5px solid #2196F3;
                padding: 20px;
                border-radius: 5px;
                margin: 10px 0;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="latest-response">
            <h4>Recipe Bot's Response:</h4>
            {latest_bot_msg.replace("/n", "<br>")}  
            </div>
            """, unsafe_allow_html=True)
    
    # Display previous chat history
    st.header("Previous Conversation History")
    if len(st.session_state.messages) > 2:
        for i in range(0, len(st.session_state.messages) - 2, 2):
            st.markdown(f"**You:** {st.session_state.messages[i]['content']}")
            st.markdown(f"**Recipe Bot:** {st.session_state.messages[i+1]['content']}")
            st.markdown("---")
    else:
        st.info("No previous conversations yet.")
    
    # Footer
    st.markdown("---")
    st.caption("Powered by Google Gemini and Streamlit • Made with ❤️ for foodies")

if __name__ == "__main__":
    main()