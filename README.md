# Motivation
[Shalit et al. (2017)](https://arxiv.org/pdf/1606.03976.pdf)の再現実験と他手法との比較を行ったレポジトリ

# setup
At first, add the following in your `~/.bashrc`.
```
function conda_cd() {
    cd $@ && [ -f ".workspace" ] && source .workspace
}
alias cd="conda_cd"
```
