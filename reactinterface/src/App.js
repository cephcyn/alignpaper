import React from 'react';

import logo from './logo.svg';
import './App.css';

class ShiftButton extends React.Component {
  constructor(props) {
    super(props);
    this.shiftButton = this.shiftButton.bind(this);
  }

  shiftButton(e) {
    e.preventDefault();
    // console.log("Shift button clicked!");
    // console.log(e);
    const requestOptions = {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        alignment: JSON.stringify(this.props.data),
        alignment_max_row_length: this.props.max_row_length,
        row: this.props.rownum,
        col: this.props.colnum,
        shift_dist: this.props.direction,
        param_score_components: this.props.param_score_components,
      })
    };
    fetch("/api/alignop/shift", requestOptions)
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        this.props.onAlignmentChange(data);
      });
  }

  render() {
    // console.log("rerendering ShiftButton ..............");
    // console.log("props:", this.props);
    // console.log("state:", this.state);

    let text;
    if (this.props.direction === -1) {
      text = "<"; // &lt;
    } else {
      text = ">"; // &gt;
    }

    return (
      <button
        className="tight"
        onClick={this.shiftButton}>
          {text}
      </button>)
  }
}

class InsertButton extends React.Component {
  constructor(props) {
    super(props);
    this.insertButton = this.insertButton.bind(this);
  }

  insertButton(e) {
    e.preventDefault();
    console.log("Insert button clicked!");
    console.log(e);
    const requestOptions = {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        alignment: JSON.stringify(this.props.data),
        alignment_max_row_length: this.props.max_row_length,
        col: this.props.colnum,
        insertafter: true,
        param_score_components: this.props.param_score_components,
      })
    };
    fetch("/api/alignop/insertcol", requestOptions)
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        this.props.onAlignmentChange(data);
      });
  }

  render() {
    // console.log("rerendering InsertButton ...........");
    // console.log("props:", this.props);
    // console.log("state:", this.state);

    return (
      <button
        className="tight"
        onClick={this.insertButton}>
          +
      </button>)
  }
}

class DeleteButton extends React.Component {
  constructor(props) {
    super(props);
    this.deleteButton = this.deleteButton.bind(this);
  }

  deleteButton(e) {
    e.preventDefault();
    console.log("Delete button clicked!");
    console.log(e);
    const requestOptions = {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        alignment: JSON.stringify(this.props.data),
        alignment_max_row_length: this.props.max_row_length,
        col: this.props.colnum,
        param_score_components: this.props.param_score_components,
      })
    };
    fetch("/api/alignop/deletecol", requestOptions)
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        this.props.onAlignmentChange(data);
      });
  }

  render() {
    // console.log("rerendering DeleteButton .............");
    // console.log("props:", this.props);
    // console.log("state:", this.state);

    return (
      <button
        className="tight"
        onClick={this.deleteButton}>
          -
      </button>)
  }
}

class MergeButton extends React.Component {
  constructor(props) {
    super(props);
    this.mergeButton = this.mergeButton.bind(this);
  }

  mergeButton(e) {
    e.preventDefault();
    console.log("Merge button clicked!");
    console.log(e);
    const requestOptions = {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        alignment: JSON.stringify(this.props.data),
        alignment_max_row_length: this.props.max_row_length,
        col: this.props.colnum,
        param_score_components: this.props.param_score_components,
      })
    };
    fetch("/api/alignop/mergecol", requestOptions)
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        this.props.onAlignmentChange(data);
      });
  }

  render() {
    // console.log("rerendering MergeButton ...........");
    // console.log("props:", this.props);
    // console.log("state:", this.state);

    return (
      <button
        className="tight"
        onClick={this.mergeButton}>
          M
      </button>)
  }
}

class SplitSingleButton extends React.Component {
  constructor(props) {
    super(props);
    this.splitSingleButton = this.splitSingleButton.bind(this);
  }

  splitSingleButton(e) {
    e.preventDefault();
    console.log("Split-Single button clicked!");
    console.log(e);
    const requestOptions = {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        alignment: JSON.stringify(this.props.data),
        alignment_max_row_length: this.props.max_row_length,
        col: this.props.colnum,
        right_align: this.props.right_align,
        param_score_components: this.props.param_score_components,
      })
    };
    fetch("/api/alignop/splitsinglecol", requestOptions)
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        this.props.onAlignmentChange(data);
      });
  }

  render() {
    // console.log("rerendering SplitSingleButton ...........");
    // console.log("props:", this.props);
    // console.log("state:", this.state);

    let contents = "SS";
    if (this.props.right_align) {
      contents += "R"
    } else {
      contents += "L"
    }

    return (
      <button
        className="tight"
        onClick={this.splitSingleButton}>
          {contents}
      </button>)
  }
}

class SplitTrieButton extends React.Component {
  constructor(props) {
    super(props);
    this.splitTrieButton = this.splitTrieButton.bind(this);
  }

  splitTrieButton(e) {
    e.preventDefault();
    console.log("Split-Trie button clicked!");
    console.log(e);
    const requestOptions = {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        alignment: JSON.stringify(this.props.data),
        alignment_max_row_length: this.props.max_row_length,
        col: this.props.colnum,
        right_align: this.props.right_align,
        param_score_components: this.props.param_score_components,
      })
    };
    fetch("/api/alignop/splittriecol", requestOptions)
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        this.props.onAlignmentChange(data);
      });
  }

  render() {
    // console.log("rerendering SplitTrieButton ...........");
    // console.log("props:", this.props);
    // console.log("state:", this.state);

    let contents = "TS";
    if (this.props.right_align) {
      contents += "R"
    } else {
      contents += "L"
    }

    return (
      <button
        className="tight"
        onClick={this.splitTrieButton}>
          {contents}
      </button>)
  }
}

class AlignmentTable extends React.Component {
  render() {
    // console.log("rerendering AlignmentTable ..............");
    // console.log("props:", this.props);
    // console.log("state:", this.state);

    const output = this.props.dataLockCols.map(
      (locked, index) => {
        return (
          <th key={index}>
            txt{index}
            <br/>
            lock
            <input
              key={"collock"+index}
              name={index}
              type="checkbox"
              checked={locked}
              onChange={this.props.handleColLockChange}
            />
            <br/>
            <InsertButton
              data={this.props.data}
              max_row_length={this.props.max_row_length}
              colnum={index}
              param_score_components={this.props.param_score_components}
              onAlignmentChange={this.props.onAlignmentChange}
            />
            <DeleteButton
              data={this.props.data}
              max_row_length={this.props.max_row_length}
              colnum={index}
              param_score_components={this.props.param_score_components}
              onAlignmentChange={this.props.onAlignmentChange}
            />
            <MergeButton
              data={this.props.data}
              max_row_length={this.props.max_row_length}
              colnum={index}
              param_score_components={this.props.param_score_components}
              onAlignmentChange={this.props.onAlignmentChange}
            />
            <br/>
            <SplitSingleButton
              data={this.props.data}
              max_row_length={this.props.max_row_length}
              colnum={index}
              right_align={false}
              param_score_components={this.props.param_score_components}
              onAlignmentChange={this.props.onAlignmentChange}
            />
            <SplitSingleButton
              data={this.props.data}
              max_row_length={this.props.max_row_length}
              colnum={index}
              right_align={true}
              param_score_components={this.props.param_score_components}
              onAlignmentChange={this.props.onAlignmentChange}
            />
            <br/>
            <SplitTrieButton
              data={this.props.data}
              max_row_length={this.props.max_row_length}
              colnum={index}
              right_align={false}
              param_score_components={this.props.param_score_components}
              onAlignmentChange={this.props.onAlignmentChange}
            />
            <SplitTrieButton
              data={this.props.data}
              max_row_length={this.props.max_row_length}
              colnum={index}
              right_align={true}
              param_score_components={this.props.param_score_components}
              onAlignmentChange={this.props.onAlignmentChange}
            />
          </th>
        );
      }
    );
    const header = (
      <tr key='header'>
        <td></td>
        {output}
      </tr>
    );

    const rows = this.props.data.map(
      (row) => {
        const cols = row.txt.map((cell, index) => {
          return (
            <td key={index}>
              <strong>{cell.join(' ')}</strong>
              <br/>
              <ShiftButton
                data={this.props.data}
                max_row_length={this.props.max_row_length}
                rownum={row.id}
                colnum={index}
                direction={-1}
                param_score_components={this.props.param_score_components}
                onAlignmentChange={this.props.onAlignmentChange}
              />
              <ShiftButton
                data={this.props.data}
                max_row_length={this.props.max_row_length}
                rownum={row.id}
                colnum={index}
                direction={1}
                param_score_components={this.props.param_score_components}
                onAlignmentChange={this.props.onAlignmentChange}
              />
            </td>
          );
        });
        return (
          <tr key={row.id}>
            <td className="align-id">{row.id}</td>
            {cols}
          </tr>
        );
      }
    );

    return (
      <table>
        <thead>
          {header}
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    );
  }
}

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      alignment: [],
      alignment_cols_locked: [],
      alignment_max_row_length: null,
      alignment_score: null,
      alignment_score_components: null,
      param_score_components: [0.2, 0.2, 1, 0, 0, 0],
      param_score_components_default: [0.2, 0.2, 1, 0, 0, 0],
      param_move_distrib: [1, 1],
      param_move_distrib_default: [1, 1],
      param_search_cutoff: 2,
      param_search_cutoff_default: 2,
      parse_constituency: {},
      inputvalue: "",
      loading: false,
      textstatus: "",
      history: [{
        // default empty values, this is ugly and redundant but I'm not going to streamline it
        alignment: [],
        alignment_cols_locked: [],
        alignment_score: null,
        alignment_score_components: null,
      }],
      history_current: 0,
    };
    this.handleTextChange = this.handleTextChange.bind(this);
    this.handleAlignmentChange = this.handleAlignmentChange.bind(this);
    this.handleColLockChange = this.handleColLockChange.bind(this);
    this.handleParamScoreComponentsChange = this.handleParamScoreComponentsChange.bind(this);
    this.handleParamMoveDistribChange = this.handleParamMoveDistribChange.bind(this);
    this.handleParamSearchCutoffChange = this.handleParamSearchCutoffChange.bind(this);
    this.alignRawText = this.alignRawText.bind(this);
    this.updateAlignmentProgress = this.updateAlignmentProgress.bind(this);
    this.alignmentScore = this.alignmentScore.bind(this);
    this.alignmentSearchButton = this.alignmentSearchButton.bind(this);
    this.alignmentSearch = this.alignmentSearch.bind(this);
    this.updateSearchProgress = this.updateSearchProgress.bind(this);
    this.buttonDoesNothing = this.buttonDoesNothing.bind(this);
    this.alignDataSave = this.alignDataSave.bind(this);
    this.alignDataLoadClick = this.alignDataLoadClick.bind(this);
    this.alignDataLoad = this.alignDataLoad.bind(this);
    this.historyUndo = this.historyUndo.bind(this);
    this.historyRedo = this.historyRedo.bind(this);
    this.historyAppend = this.historyAppend.bind(this);
    this.historyReset = this.historyReset.bind(this);
  }

  handleTextChange(e) {
    this.setState({ inputvalue: e.target.value });
  }

  handleAlignmentChange(e) {
    // console.log('in handleAlignmentChange');
    let updated_alignment_cols_locked = this.state.alignment_cols_locked;
    if (
      (e.alignment.length > 0)
      && (e.alignment[0]['txt'].length !== this.state.alignment[0]['txt'].length)
    ) {
      updated_alignment_cols_locked = new Array(e.alignment[0]['txt'].length).fill(false);
      this.setState({ alignment_cols_locked: updated_alignment_cols_locked });
    }
    this.setState({ alignment: e.alignment });
    let updated_alignment_score = this.state.alignment_score;
    if ('alignment_score' in e) {
      updated_alignment_score = e.alignment_score;
      this.setState({ alignment_score: updated_alignment_score });
    }
    let updated_alignment_score_components = this.state.alignment_score_components;
    if ('alignment_score_components' in e) {
      updated_alignment_score_components = e.alignment_score_components;
      this.setState({ alignment_score_components: updated_alignment_score_components });
    }
    // // automatically get the new alignment score and components
    // this.alignmentScore(e);

    // update history
    this.historyAppend({
      alignment: e.alignment,
      alignment_cols_locked: updated_alignment_cols_locked,
      alignment_score: updated_alignment_score,
      alignment_score_components: updated_alignment_score_components
    });
  }

  handleColLockChange(e) {
    // console.log('in handleColLockChange');
    this.setState((prevState, props) => {
      // this is an awful hack for deep cloning this list
      let updated = JSON.parse(JSON.stringify(prevState.alignment_cols_locked));
      updated[e.target.name] = !updated[e.target.name];

      // update history
      this.historyAppend({
        alignment: this.state.alignment,
        alignment_cols_locked: updated,
        alignment_score: this.state.alignment_score,
        alignment_score_components: this.state.alignment_score_components
      });

      return { alignment_cols_locked: updated };
    });


  }

  handleParamScoreComponentsChange(e, paramidx) {
    let modified = JSON.parse(JSON.stringify(this.state.param_score_components));
    modified[paramidx] = e.target.value;
    this.setState({ param_score_components: modified });
  }

  handleParamMoveDistribChange(e, paramidx) {
    let modified = JSON.parse(JSON.stringify(this.state.param_move_distrib));
    modified[paramidx] = e.target.value;
    this.setState({ param_move_distrib: modified });
  }

  handleParamSearchCutoffChange(e) {
    this.setState({ param_search_cutoff: e.target.value });
  }

  alignRawText(e) {
    e.preventDefault();
    console.log("Raw text align button clicked!");
    this.setState({ loading: true });
    const requestOptions = {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        input: this.state.inputvalue,
        param_score_components: this.state.param_score_components,
      })
    };
    // clear history when we submit a raw alignment request
    this.historyReset();
    fetch("/api/textalign", requestOptions)
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        this.updateAlignmentProgress(data['location']);
      });
  }

  updateAlignmentProgress(status_url) {
    fetch(status_url)
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        console.log('got update from server...');
        console.log(data);
        if (data['state'] !== 'PENDING' && data['state'] !== 'PROGRESS') {
          if ('alignment' in data) {
            // success!
            this.setState({
              alignment: data['alignment'],
              // when doing initial alignment, no columns are locked
              alignment_cols_locked: new Array(data['alignment'][0]['txt'].length).fill(false),
              alignment_score: data['alignment_score'],
              alignment_score_components: data['alignment_score_components'],
              alignment_max_row_length: data['alignment_max_row_length'],
              parse_constituency: data['parse_constituency'],
              loading: false,
              textstatus: "",
            });
            // and automatically chain in an alignment search to finetune it a bit...
            this.alignmentSearch(null, 10);
          } else {
            // failure?
            this.setState({
              alignment: [],
              parse_constituency: {},
              loading: false,
              textstatus: data['status'],
            });
          }
        } else {
          // check back on the progress every so often...
          this.setState({ textstatus: data['status'] });
          setTimeout(() => {
            this.updateAlignmentProgress(status_url);
          }, 1000);
        }
      });
  }

  alignmentScore(e) {
    try {e.preventDefault();} catch {}
    this.setState({ loading: true });
    const requestOptions = {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        alignment: JSON.stringify(this.state.alignment),
        alignment_max_row_length: this.state.alignment_max_row_length,
        param_score_components: this.state.param_score_components,
      })
    };
    fetch("/api/alignscore", requestOptions)
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        this.setState({ loading: false });
        this.setState(data);
      });
  }

  alignmentSearchButton(e, numsteps) {
    e.preventDefault();
    console.log("alignment search button clicked!");
    this.alignmentSearch(e, numsteps);
  }

  alignmentSearch(e, numsteps) {
    console.log("performing alignment search!");
    this.setState({ loading: true });
    const requestOptions = {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        alignment: JSON.stringify(this.state.alignment),
        alignment_cols_locked: JSON.stringify(this.state.alignment_cols_locked),
        alignment_max_row_length: this.state.alignment_max_row_length,
        greedysteps: JSON.stringify(numsteps),
        param_score_components: this.state.param_score_components,
        param_move_distrib: this.state.param_move_distrib,
        param_search_cutoff: JSON.stringify(this.state.param_search_cutoff),
      })
    };
    fetch("/api/alignsearch", requestOptions)
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        this.updateSearchProgress(data['location']);
      });
  }

  updateSearchProgress(status_url) {
    fetch(status_url)
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        console.log('got update from server...');
        console.log(data);
        if (data['state'] !== 'PENDING' && data['state'] !== 'PROGRESS') {
          if ('alignment' in data) {
            // success!
            this.setState({
              alignment: data['alignment'],
              alignment_score: data['alignment_score'],
              alignment_score_components: data['alignment_score_components'],
              loading: false,
              textstatus: data['status'],
            });

            // update history
            this.historyAppend({
              alignment: data['alignment'],
              alignment_cols_locked: this.state.alignment_cols_locked,
              alignment_score: data['alignment_score'],
              alignment_score_components: data['alignment_score_components']
            });
          } else {
            // failure?
            this.setState({
              alignment: [],
              loading: false,
              textstatus: data['status']
            });
          }
        } else {
          // check back on the progress every so often...
          this.setState({ textstatus: data['status'] });
          setTimeout(() => {
            this.updateSearchProgress(status_url);
          }, 1000);
        }
      });
  }

  buttonDoesNothing(e) {
    e.preventDefault();
    console.log("nothing button clicked!");
  }

  alignDataSave(e) {
    e.preventDefault();
    console.log("save button clicked!");
    const output = JSON.stringify({
      alignment: this.state.alignment,
      alignment_cols_locked: this.state.alignment_cols_locked,
      alignment_score: this.state.alignment_score,
      alignment_score_components: this.state_alignment_score_components,
      alignment_max_row_length: this.state.alignment_max_row_length,
      parse_constituency: this.state.parse_constituency,
    });
    const blob = new Blob([output]);
    const fileDownloadUrl = URL.createObjectURL(blob);
    this.setState ({fileDownloadUrl: fileDownloadUrl},
      () => {
        this.dofileDownload.click();
        URL.revokeObjectURL(fileDownloadUrl);  // free up storage--no longer needed.
        this.setState({fileDownloadUrl: ""})
      }
    );
  }

  alignDataLoadClick(e) {
    e.preventDefault();
    console.log("load button clicked!");
    this.dofileUpload.click()
  }

  alignDataLoad(e) {
    const fileObj = e.target.files[0];
    const reader = new FileReader();
    let fileloaded = e => {
      // e.target.result is the file's content as text
      const fileContents = e.target.result;
      const fileContentsParse = JSON.parse(fileContents);
      this.setState(fileContentsParse);
      this.setState({
        textstatus: 'Loaded alignment from save file',
      });

      // set history purely to loaded contents when we load in from a saved alignment
      this.historyReset({
        alignment: fileContentsParse.alignment,
        alignment_cols_locked: fileContentsParse.alignment_cols_locked,
        alignment_score: fileContentsParse.alignment_score,
        alignment_score_components: fileContentsParse.alignment_score_components
      });
    }

    fileloaded = fileloaded.bind(this);
    reader.onload = fileloaded;
    reader.readAsText(fileObj);
  }

  historyUndo(e) {
    e.preventDefault();
    console.log("undo button clicked!");
    this.setState({
      alignment: this.state.history[this.state.history_current - 1].alignment,
      alignment_cols_locked: this.state.history[this.state.history_current - 1].alignment_cols_locked,
      alignment_score: this.state.history[this.state.history_current - 1].alignment_score,
      alignment_score_components: this.state.history[this.state.history_current - 1].alignment_score_components,
      history_current: this.state.history_current - 1
    });
  }

  historyRedo(e) {
    e.preventDefault();
    console.log("redo button clicked!");
    this.setState({
      alignment: this.state.history[this.state.history_current + 1].alignment,
      alignment_cols_locked: this.state.history[this.state.history_current + 1].alignment_cols_locked,
      alignment_score: this.state.history[this.state.history_current + 1].alignment_score,
      alignment_score_components: this.state.history[this.state.history_current + 1].alignment_score_components,
      history_current: this.state.history_current + 1
    });
  }

  historyAppend(checkpoint) {
    console.log("attempting to append to history!");
    // first check if an append actually should be done
    // e.g. if a shift makes no change to the alignment at all, there should be no step added to history
    // TODO implement the check
    if (false) {
      // we don't need to do an append...
      console.log("found that no append was necessary!");
    } else {
      // actually do the append and update current index tracker
      if (this.state.history.length === this.state.history_current) {
        // we are at the tail end of history, just append
        this.setState({
          history: this.state.history.concat([checkpoint]),
          history_current: this.state.history_current + 1
        });
      } else {
        // we are in the midpoint of history, cut off the tail bit first
        this.setState({
          history: this.state.history.slice(0, this.state.history_current + 1).concat([checkpoint]),
          history_current: this.state.history_current + 1
        });
      }
    }
  }

  historyReset(checkpoint) {
    console.log("resetting history!");
    // input value is the state to reset to; if non-null then use that state
    if (checkpoint) {
      this.setState({
        history: [checkpoint]
      });
    } else {
      // reset to empty state
      this.setState({
        history: [{
          alignment: [],
          alignment_cols_locked: [],
          alignment_score: null,
          alignment_score_components: null,
        }]
      });
    }
    this.setState({ history_current: 0 });
  }

  render() {
    console.log("rerendering App.......", new Date());
    console.log("state:", this.state);

    // only render loading indicator if we are currently waiting on the api
    let loadingspinner;
    if (this.state.loading) {
      loadingspinner = <p>Working...</p>
    } else {
      loadingspinner = <br />
    }

    // build the alignment table
    let aligntable;
    if (this.state.alignment.length > 0) {
      aligntable = <AlignmentTable
        data={this.state.alignment}
        max_row_length={this.state.alignment_max_row_length}
        param_score_components={this.state.param_score_components}
        dataLockCols={this.state.alignment_cols_locked}
        onAlignmentChange={this.handleAlignmentChange}
        handleColLockChange={this.handleColLockChange}
      />
    } else {
      aligntable = <br />
    }

    // build the score component weighting hyperparameter control table
    let scorecomponenttable = [
      "alignment length",
      "column filled-ness",
      "column agreement",
      "distinct tokens",
      "distinct entity TUIs",
      "term column count",
    ].map((component_name, index) => {
      return (
        <tr key={index}>
          <td>
            {component_name}
          </td>
          <td>
            <input
              type="range"
              min="-1"
              max="1"
              step="0.1"
              value={this.state.param_score_components[index]}
              onChange={e => this.handleParamScoreComponentsChange(e, index)}
            />
          </td>
          <td>
            {this.state.param_score_components[index]}
          </td>
        </tr>
      );
    });
    scorecomponenttable = (
      <table>
        <tbody>
          {scorecomponenttable}
        </tbody>
      </table>
    );

    // build the search step probability hyperparameter control table
    let movedistribtable = [
      "greedy",
      "random",
    ].map((move_name, index) => {
      return (
        <tr key={index}>
          <td>
            {move_name}
          </td>
          <td>
            <input
              type="range"
              min="0"
              max="10"
              step="1"
              value={this.state.param_move_distrib[index]}
              onChange={e => this.handleParamMoveDistribChange(e, index)}
            />
          </td>
          <td>
            {this.state.param_move_distrib[index]}
          </td>
        </tr>
      );
    });
    movedistribtable = (
      <table>
        <tbody>
          {movedistribtable}
        </tbody>
      </table>
    );

    // build the search step cutoff hyperparameter table
    let searchcutofftable = [
      "greedy step cutoff",
    ].map((component_name, index) => {
      return (
        <tr key={index}>
          <td>
            {component_name}
          </td>
          <td>
            <input
              type="range"
              min="1"
              max="20"
              step="1"
              value={this.state.param_search_cutoff}
              onChange={e => this.handleParamSearchCutoffChange(e)}
            />
          </td>
          <td>
            {this.state.param_search_cutoff}
          </td>
        </tr>
      );
    });
    searchcutofftable = (
      <table>
        <tbody>
          {searchcutofftable}
        </tbody>
      </table>
    );

    // do checks to see if undo and redo are legal with the history and current state
    let historyUndoLegal = this.state.history_current > 0;
    let historyRedoLegal = this.state.history.length > this.state.history_current + 1;

    return (
      <div className="App">
        <textarea
          value={this.state.inputvalue}
          onChange={this.handleTextChange}
          className="raw-input"
        />
        <br />
        <button onClick={this.alignRawText}>Align Texts</button>
        <button onClick={this.alignmentScore}>Score</button>
        <br />
        <button onClick={e => this.alignmentSearchButton(e, 1)}>Search (1 step)</button>
        <button onClick={e => this.alignmentSearchButton(e, 10)}>Search (up to 10 steps)</button>
        <button onClick={e => this.alignmentSearchButton(e, 50)}>Search (up to 50 steps)</button>
        <br />
        <button onClick={this.historyUndo} disabled={!historyUndoLegal}>Undo</button>
        <button onClick={this.historyRedo} disabled={!historyRedoLegal}>Redo</button>
        <br />
        <button onClick={this.alignDataSave}>Save Alignment</button>
        <a className="hidden"
           download="alignment.json"
           href={this.state.fileDownloadUrl}
           ref={e=>this.dofileDownload = e}
        >download-href</a>
        <button onClick={this.alignDataLoadClick}>Load Alignment</button>
        <input type="file" className="hidden"
            multiple={false}
            accept=".json,.text,application/json"
            onChange={e => this.alignDataLoad(e)}
            ref={e=>this.dofileUpload = e}
          />
        <button onClick={this.buttonDoesNothing}>This Button Does Nothing</button>
        <br />
        <br />
        <hr />
        {aligntable}
        {loadingspinner}
        <p className="preservenewline">{this.state.textstatus}</p>
        <hr />
        <table style={{width: "100%"}}>
          <tbody><tr>
            <td>
              <p>overall alignment score = {this.state.alignment_score ? this.state.alignment_score.toString() : 'Undefined'}</p>
              <p>score components breakdown = {this.state.alignment_score_components ? this.state.alignment_score_components.toString() : 'Undefined'}</p>
              <p>longest single input = {this.state.alignment_max_row_length ? this.state.alignment_max_row_length.toString() : 'Undefined'}</p>
            </td>
            <td>
              {scorecomponenttable}
              {movedistribtable}
              {searchcutofftable}
            </td>
          </tr></tbody>
        </table>
        <hr />
        <img src={logo} className="App-logo" alt="logo" />
      </div>
    );
  }
}

export default App;
