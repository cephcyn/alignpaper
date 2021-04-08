import React from 'react';

import logo from './logo.svg';
import './App.css';

class AlignmentTable extends React.Component {
  render() {
    // console.log("rerendering AlignmentTable =========");
    // console.log("props:", this.props);
    // console.log("state:", this.state);

    const rows = this.props.data.map(
      (row) => {
        const cols = row.txt.map((cell, index) => {
          return <td key={index}>{cell.join(' ')}</td>
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
      inputvalue: "",
      loading: false,
    };

    this.handleChange = this.handleChange.bind(this);
    this.activateLasers = this.activateLasers.bind(this);
  }

  componentDidMount() {}

  handleChange(e) {
    this.setState({inputvalue: e.target.value});
  }

  activateLasers(e) {
    e.preventDefault();
    console.log("Button clicked!");
    console.log(e);
    console.log("value=");
    console.log(this.state.inputvalue);
    this.setState({ loading: true });
    fetch(this.props.apiUrl+new URLSearchParams({
      input: this.state.inputvalue,
      // id: "3",
    }))
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        this.setState({ loading: false });
        this.setState(data);
      });
  }

  render() {
    console.log("rerendering App =========", new Date());
    console.log("props:", this.props);
    console.log("state:", this.state);

    // only render waiting spinner if we are currently waiting on the api
    let spinner;
    if (this.state.loading) {
      spinner = <p>Loading...</p>
    } else {
      spinner = <br/>
    }

    // only render alignment if there's content
    let aligntable;
    if (this.state.alignment.length > 0) {
      aligntable = <AlignmentTable data={this.state.alignment} />
    } else {
      aligntable = <br />
    }

    return (
      <div className="App">
        <textarea value={this.state.inputvalue} onChange={this.handleChange} />
        <br />
        <button onClick={this.activateLasers}>Submit Request</button>
        <br />
        <br />
        {aligntable}
        {spinner}
        <p>temp_arg_input is...</p>
        <p>{this.state.temp_arg_input ? this.state.temp_arg_input.toString() : 'Undefined'}</p>
        <br />
        <img src={logo} className="App-logo" alt="logo" />
      </div>
    );
  }
}

export default App;
