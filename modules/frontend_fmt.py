import streamlit as st
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb

def html_header(url):
     st.markdown(f'<b style="color:#439AD6;font-size:32px;">{url}</b>', unsafe_allow_html=True)

def html_txt(url):
     st.markdown(f'<p style="background-color:#FFFFFF;color:#384B8F;">{url}</p>', unsafe_allow_html=True)

def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))

def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)

def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp {bottom: 80px;}
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="#2E5987",
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(-70, 0, 10, 0),
        border_style="inset",
        border_width=px(1)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        "Designed in ",
        link('https://streamlit.io/',
            image('https://avatars3.githubusercontent.com/u/45109972?s=400&v=4',
                  width=px(25), height=px(25))),
        " by Carlos Espejo Peña",
        br(),
        link("https://www.linkedin.com/in/carlosespejopena/", image('https://drive.google.com/uc?export=view&id=1nx0u9GeUyYttqyju6Z1824UCqto6hXZv')),
        " | ",
        link("https://github.com/caresppen", image('https://drive.google.com/uc?export=view&id=17_77FAziJKdyZaRkjzlGFTKaPAKGdszl')),
    ]
    layout(*myargs)

    
def more_children_or_more_pets_background(row):    

    highlight = 'background-color: lightcoral;'
    default = ''

    # must return one string per cell in this row
    if row['num_children'] > row['num_pets']:
        return [highlight, default]
    elif row['num_pets'] > row['num_children']:
        return [default, highlight]
    else:
        return [default, default]
