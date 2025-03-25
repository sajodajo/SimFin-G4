import streamlit as st

st.set_page_config(
    page_title="About the Team",
    page_icon='Media/flexiTradeIcon.png',
    layout = 'wide'
)


st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
        }
    </style>
    """, unsafe_allow_html=True)


col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("Media/ftTeam.png", width=800)

st.markdown("""
<div style="text-align: center; font-size: 18px; line-height: 1.6; max-width: 1350px; margin: 0 auto;">
<p><strong>FlexiTrade</strong> is an automated daily trading tool designed and built by a small, focused team of engineers, data scientists, and market enthusiasts. We come from diverse professional backgrounds — spanning finance, machine learning, and software development — but share a common goal: to simplify and strengthen the way automated trading is done.</p>

<p>What sets our team apart is a hands-on approach. We’re not just building tools — we’re actively testing strategies, fine-tuning models, and iterating on systems to ensure they hold up in real-world conditions. Every feature in FlexiTrade is the result of direct collaboration across disciplines, combining technical rigor with practical trading insight.<br><br></p>
</div>
""", unsafe_allow_html=True)


# Create 5 columns
col1, col2, col3, col4, col5 = st.columns(5)


linkedinPic = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQVwpVlXEdO4SYUobUhd4C0DxLhsil1mNLpiw&s'
sjLI = 'https://www.linkedin.com/in/sajodajo/'
jmLI = 'https://www.linkedin.com/in/joaqu%C3%ADn-mi%C3%B1o-perez-65049b20b/'
ysLI = 'https://www.linkedin.com/in/yeabsira-seleshi-14aa8b27a/'
wbLI = 'https://www.linkedin.com/in/waldo-barreto-tascon-4bb857292/'
aoLI = 'https://www.linkedin.com/in/alejandroosto/'

with col1:
    st.image('https://media.licdn.com/dms/image/v2/D4D03AQEBnWkkwXXlzA/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1719831432445?e=1748476800&v=beta&t=uR3F0SZgHl5n0ygvizzFS6kYSbeWHLNj6evb2bgOykc', use_container_width=True)
    st.subheader(' 🇮🇪 Sam Jones')
    st.write('MSc student of Data Science at IE University, with two years of experience in innovation consulting focused on sustainability, digital technologies, and startup engagement. Skilled in Python and SQL for data analytics and database management, with a solid understanding of machine learning, AI, and modern data architectures.')
    st.markdown(
    f'''
    <div style="text-align: center;">
        <a href="{sjLI}" target="_blank">
            <img src="{linkedinPic}" style="max-width:75px; height:auto;" />
        </a>
    </div>
    ''',
    unsafe_allow_html=True
    )

# Column 2
with col2:
    st.image('https://media.licdn.com/dms/image/v2/D4E03AQH80GUxmNjL_A/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1731686881859?e=1748476800&v=beta&t=lOy7y2jiQgXPhczHMRym49gUxS-AzCh9Ct0Ar3iCNI0', use_container_width=True)
    st.subheader(' 🇪🇨 Joaquin Miño')
    st.write('Motivated professional from Ecuador with a strong foundation in sales and data analytics. With three years of experience at an insurance broker in the sales department, excels in strategic client engagement. Currently finishing the master’s in Business Analytics and Big Data at IE, skilled in SQL, Python and Machine Learning.')
    st.markdown(
    f'''
    <div style="text-align: center;">
        <a href="{jmLI}" target="_blank">
            <img src="{linkedinPic}" style="max-width:75px; height:auto;" />
        </a>
    </div>
    ''',
    unsafe_allow_html=True
    )

with col3:
    st.image('Media/yeabsira.jpeg', use_container_width=True)
    st.subheader(' 🇪🇹 Yeabsira Seleshi')
    st.write('Results-driven Data Engineer with telecom and IT experience in network infrastructure, data pipelines, and workflow optimization. Skilled in Python, SQL, MongoDB, Spark, and data visualization (Power BI, Tableau). Currently pursuing a Master’s in Business Analytics and Data Science at IE University in Madrid, Spain.')
    st.markdown(
    f'''
    <div style="text-align: center;">
        <a href="{ysLI}" target="_blank">
            <img src="{linkedinPic}" style="max-width:75px; height:auto;" />
        </a>
    </div>
    ''',
    unsafe_allow_html=True
    )
    
# Column 4
with col4:
    st.image('Media/waldo.jpeg', use_container_width=True)
    st.subheader(' 🇪🇸 Waldo Barreto')
    st.write("Economist currently pursuing a master's in Data Science and Business Analytics at IE University in Madrid, Spain, with a strong work ethic, and thrive on leveraging analytical skills to solve complex business problems. Deeply committed to continuous learning and growth, always seeking new opportunities to expand expertise.")
    st.markdown(
    f'''
    <div style="text-align: center;">
        <a href="{wbLI}" target="_blank">
            <img src="{linkedinPic}" style="max-width:75px; height:auto;" />
        </a>
    </div>
    ''',
    unsafe_allow_html=True
    )
    
# Column 5
with col5:
    st.image('https://media.licdn.com/dms/image/v2/D4E03AQGBfBq6aCqPFQ/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1731001763098?e=1748476800&v=beta&t=moDnGyQyRicp-Lezf8pOJSxN8aLasjvKSKWbubw16FM', use_container_width=True)
    st.subheader(' 🇵🇹 Alejandro Osto')
    st.write('Finance graduate with experience in Finance, Banking and Recruitment, with strong skills in Excel, SQL, and Python. Roles in banking, consulting, and recruitment have provided valuable insights into strategic client engagement. Working with companies like BNP Paribas has built strong leadership and communication abilities.')
    st.markdown(
    f'''
    <div style="text-align: center;">
        <a href="{aoLI}" target="_blank">
            <img src="{linkedinPic}" style="max-width:75px; height:auto;" />
        </a>
    </div>
    ''',
    unsafe_allow_html=True
    )