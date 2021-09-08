# Motivation
This project is for a experiment to reproduce the result of [Shalit et al. (2017)](https://arxiv.org/pdf/1606.03976.pdf).

# setup
At first, add the following in your `~/.bashrc`.
```
function conda_cd() {
    cd $@ && [ -f ".workspace" ] && source .workspace
}
alias cd="conda_cd"
```
