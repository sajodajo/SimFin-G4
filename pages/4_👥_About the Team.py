import streamlit as st

st.set_page_config(
    page_title="About the Team",
    page_icon="👥",
    layout = 'wide'
)

st.title(f'💫 About the Team\n\n')


import streamlit as st




st.write('''At Stock Insights Hub, we are proud to be a dynamic, global team bringing a wealth of knowledge and expertise from across the world. Our diverse backgrounds and experiences empower us to innovate and provide unique solutions for stock market analysis, prediction, and strategy optimization. Together, we strive to create cutting-edge tools that enable our users to succeed in the fast-paced world of finance.

Our team consists of experts from a wide range of disciplines, from finance and machine learning to software development and strategy. Each member contributes their own perspective, allowing us to tackle challenges from multiple angles and continuously improve our platform.''')


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
    st.subheader('Waldo Barreto Tascon')
    st.write('WB Description')

# Column 5
with col5:
    st.image('https://media.licdn.com/dms/image/v2/D4E03AQGBfBq6aCqPFQ/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1731001763098?e=1748476800&v=beta&t=moDnGyQyRicp-Lezf8pOJSxN8aLasjvKSKWbubw16FM', use_container_width=True)
    st.subheader('Alejandro Osto')
    st.write('AO Description')