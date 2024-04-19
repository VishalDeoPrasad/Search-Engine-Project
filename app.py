import streamlit as st

def main():
    # Set page title and favicon
    st.set_page_config(
        page_title="String Input Example",
        page_icon=":pencil2:"
    )

    # Title and subtitle
    st.title("String Input Example")
    st.markdown("*Enter a string below and see it displayed:*")

    # Text input widget with placeholder
    user_input = st.text_input("Enter a string:", "")

    # Display the entered string
    if user_input:
        st.markdown(f"**You entered:** `{user_input}`")

if __name__ == "__main__":
    main()
