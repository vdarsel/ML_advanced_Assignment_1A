from EMfunctions import *
import plotly.graph_objects as go

def main():
    X_1,X_2,S = read_files("X.txt","S.txt")
    fig = go.Figure(data=go.Scatter(x=X_1,y=X_2,mode="markers", marker={"color":S,'coloraxis':'coloraxis'}),
                 layout={"xaxis_title":"X 1","yaxis_title":"X 2","legend_title":"S", "title":"Distribution", "coloraxis_colorbar_title_text":"S"})
    fig.write_html('example.html', auto_open=True)
    fig = go.Figure(data=go.Scatter3d(x=X_1,y=S,z=X_2,mode="markers"))
                # layout={"xaxis_title":,"yaxis_title":"S","zaxis_title":"X 2", "title":"Distribution"})
    fig.update_layout(scene = dict(xaxis_title = "X 1",yaxis_title = "S",zaxis_title = "X 2"),title="Distribution")
    fig.write_html('example2.html', auto_open=True)
    max_val = 8
    val = np.arange(1,max_val)
    BIC= np.zeros(len(val))
    for k in val:
        mu_1,mu_2,tau_1,tau_2,lambd,pi = EM(0.1,k,X_1,X_2,S)
        proba = np.zeros(len(X_1))-1
        for i in range(len(X_1)):
            temp = proba_point(X_1[i],X_2[i],S[i],mu_1,mu_2,tau_1,tau_2,lambd,pi)
            proba[i] = np.sum(temp)
        BIC[k-val[0]] = (6*k)*np.log(len(X_1))- 2*np.sum(np.log(proba))
    plt.figure(figsize=(10,7))
    plt.plot(val,BIC)
    plt.xlabel("value of k")
    plt.ylabel("BIC")
    plt.show()
    k = np.argmin(BIC)+1
    mu_1,mu_2,tau_1,tau_2,lambd,pi = EM(0.05,k,X_1,X_2,S)
    aff = affectation(X_1,X_2,S,mu_1,mu_2,tau_1,tau_2,lambd,pi)
    fig = go.Figure(data=go.Scatter3d(x=X_1,y=S,z=X_2,mode="markers",marker={"color":aff, 'coloraxis':'coloraxis'} ),)
    fig.update_layout(scene = dict(xaxis_title = "X 1",yaxis_title = "S",zaxis_title = "X 2"),title="Most likely cluster representation for best k (k="+str(k)+")")
    fig.write_html('example.html', auto_open=True)

if __name__ == "__main__":
    main()