from EMfunctions import *
import plotly.graph_objects as go

def main():
    X_1,X_2,S = read_files("X.txt","S.txt")
    fig = go.Figure(data=go.Scatter(x=X_1,y=X_2,mode="markers", marker={"color":S,'coloraxis':'coloraxis'}))
    fig.write_html('example.html', auto_open=True)
    fig = go.Figure(data=go.Scatter3d(x=X_1,y=S,z=X_2,mode="markers"))
                # layout={"xaxis_title":,"yaxis_title":"S","zaxis_title":"X 2", "title":"Distribution"})
    fig.update_layout(scene = dict(xaxis_title = "X 1",yaxis_title = "S",zaxis_title = "X 2"),title="Distribution")
    fig.write_html('example2.html', auto_open=True)
    mu_1,mu_2,tau_1,tau_2,lambd,pi = EM(0.05,3,X_1,X_2,S)
    aff = affectation(X_1,X_2,S,mu_1,mu_2,tau_1,tau_2,lambd,pi)
    fig = go.Figure(data=go.Scatter3d(x=X_1,y=S,z=X_2,mode="markers",marker={"color":aff, 'coloraxis':'coloraxis'}),)
    fig.update_layout(scene = dict(xaxis_title = "X 1",yaxis_title = "S",zaxis_title = "X 2"),title="Most likely cluster representation k=3")
    fig.write_html('example.html', auto_open=True)
    mu_1,mu_2,tau_1,tau_2,lambd,pi = EM(0.1,5,X_1,X_2,S)
    aff = affectation(X_1,X_2,S,mu_1,mu_2,tau_1,tau_2,lambd,pi)
    fig = go.Figure(data=go.Scatter3d(x=X_1,y=S,z=X_2,mode="markers",marker={"color":aff, 'coloraxis':'coloraxis'}),)
    fig.update_layout(scene = dict(xaxis_title = "X 1",yaxis_title = "S",zaxis_title = "X 2"),title="Most likely cluster representation k=5")
    fig.write_html('example.html', auto_open=True)


if __name__ == "__main__":
    main()