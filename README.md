# 動機
This project is for a replicational experiment of [Shalit et al. (2017)](https://arxiv.org/pdf/1606.03976.pdf).

# setup
At first, add the following in your `~/.bashrc`.
```
function conda_cd() {
    cd $@ && [ -f ".workspace" ] && source .workspace
}
alias cd="conda_cd"
```
