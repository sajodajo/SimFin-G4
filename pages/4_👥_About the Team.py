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


st.title(f'The FlexiTrade Team\n\n')


st.markdown("""
<div style="text-align: center; font-size: 18px; line-height: 1.6; max-width: 1350px; margin: 0 auto;">
<p><strong>FlexiTrade</strong> is an automated daily trading tool designed and built by a small, focused team of engineers, data scientists, and market enthusiasts. We come from diverse professional backgrounds — spanning finance, machine learning, and software development — but share a common goal: to simplify and strengthen the way automated trading is done.</p>

<p>What sets our team apart is a hands-on approach. We’re not just building tools — we’re actively testing strategies, fine-tuning models, and iterating on systems to ensure they hold up in real-world conditions. Every feature in FlexiTrade is the result of direct collaboration across disciplines, combining technical rigor with practical trading insight.<br><br></p>
</div>
""", unsafe_allow_html=True)


# Create 5 columns
col1, col2, col3, col4, col5 = st.columns(5)


# Column 1
with col1:
    st.image('https://media.licdn.com/dms/image/v2/D4D03AQEBnWkkwXXlzA/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1719831432445?e=1748476800&v=beta&t=uR3F0SZgHl5n0ygvizzFS6kYSbeWHLNj6evb2bgOykc', use_container_width=True)
    st.subheader('Sam Jones')
    st.write('SJ Description')

# Column 2
with col2:
    st.image('https://media.licdn.com/dms/image/v2/D4E03AQH80GUxmNjL_A/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1731686881859?e=1748476800&v=beta&t=lOy7y2jiQgXPhczHMRym49gUxS-AzCh9Ct0Ar3iCNI0', use_container_width=True)
    st.subheader('Joaquin Miño')
    st.write('JM Description')

# Column 3
with col3:
    st.image('https://media.licdn.com/dms/image/v2/D4D35AQHF38k5qgpIpg/profile-framedphoto-shrink_400_400/B4DZWL6pXuHAAg-/0/1741809162134?e=1743354000&v=beta&t=nKm5pmPA0z7RuSU6UqnU5ky8ZXNh5nsRPoJ6_LDM0L8', use_container_width=True)
    st.subheader('Yeabsira Seleshi')
    st.write('YS Description')

# Column 4
with col4:
    st.image('https://media.licdn.com/dms/image/v2/D4E03AQHXn-PMAQP0IQ/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1695490854107?e=1748476800&v=beta&t=eBbigPp61ssoLheKLO7YlPliMAywKRzxSSFty98y_os', use_container_width=True)
    st.subheader('Waldo Barreto')
    st.write('WB Description')

# Column 5
with col5:
    st.image('https://media.licdn.com/dms/image/v2/D4E03AQGBfBq6aCqPFQ/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1731001763098?e=1748476800&v=beta&t=moDnGyQyRicp-Lezf8pOJSxN8aLasjvKSKWbubw16FM', use_container_width=True)
    st.subheader('Alejandro Osto')
    st.write('AO Description')