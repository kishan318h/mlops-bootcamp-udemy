## github-actions-demo
Github actions is an integrated automation platform for Github which allows us to automate, customize and execute our workflow directly in ur repository.
The commands for automation are written in a YAML file. This file should be present in `root/.github/workflows`. Root directory is the repository. The automation will not work if `.github` folder is present inside a sub-directory.

**Note**: `.github` has been moved to a subfolder.


## Example code for a YAML file

```YAML
name: GitHub Actions Demo  # name of the workflow
run-name: ${{ github.actor }} is testing out GitHub Actions 🚀
on: [push] #  run the automation when the file is pushed to repo
jobs:
  Explore-GitHub-Actions:
    runs-on: ubuntu-latest # the actions/executions will be perfromed on latest version of ubuntu hosted on Github
    steps:
      - run: echo "🎉 The job was automatically triggered by a ${{ github.event_name }} event."
      - run: echo "🐧 This job is now running on a ${{ runner.os }} server hosted by GitHub!"
      - run: echo "🔎 The name of your branch is ${{ github.ref }} and your repository is ${{ github.repository }}."
      - name: Check out repository code
        uses: actions/checkout@v4 # copy the repository into runner workspace
      - run: echo "💡 The ${{ github.repository }} repository has been cloned to the runner."
      - run: echo "🖥️ The workflow is now ready to test your code on the runner."
      - name: List files in the repository
        run: |
          ls ${{ github.workspace }}
      - run: echo "🍏 This job's status is ${{ job.status }}."

# the runner workspace is deleted after the execution if the workspace in on Github-hosted runners
```

## Create workflow
Github have template for a lot of workflow. To create a workflow, click on action > workflow and choose the relevant one for the task. Edit the template based on project requirements.